#include "face_emotion_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <sstream>

#include <util/platform.h>

#include <opencv2/imgproc.hpp>

#include <plugin-support.h>

#include "emotion_mapping.hpp"

namespace {

constexpr const char *kFilterId = "face_emotion_filter";
constexpr const char *kFaceModelName = "face_detection_yunet_2023mar.onnx";
constexpr const char *kEmotionModelName = "emotion-ferplus-8.onnx";

constexpr const char *kSettingEnabled = "enabled";
constexpr const char *kSettingMaxFaces = "max_faces";
constexpr const char *kSettingInferenceFps = "inference_fps";
constexpr const char *kSettingInferenceWidth = "inference_width";
constexpr const char *kSettingConfidenceThreshold = "confidence_threshold";
constexpr const char *kSettingSmoothingSeconds = "smoothing_seconds";
constexpr const char *kSettingShowConfidence = "show_confidence";
constexpr const char *kSettingShowBox = "show_box";
constexpr const char *kSettingShowLabel = "show_label";
constexpr const char *kSettingBoxUseEmotionColor = "box_use_emotion_color";
constexpr const char *kSettingBoxColorR = "box_color_r";
constexpr const char *kSettingBoxColorG = "box_color_g";
constexpr const char *kSettingBoxColorB = "box_color_b";
constexpr const char *kSettingBoxThickness = "box_thickness";
constexpr const char *kSettingShowTrackId = "show_track_id";
constexpr const char *kSettingTextUseEmotionColor = "text_use_emotion_color";
constexpr const char *kSettingTextColorR = "text_color_r";
constexpr const char *kSettingTextColorG = "text_color_g";
constexpr const char *kSettingTextColorB = "text_color_b";
constexpr const char *kSettingTextOpacity = "text_opacity";
constexpr const char *kSettingTextScale = "text_scale";
constexpr const char *kSettingTextThickness = "text_thickness";
constexpr const char *kSettingTextPadding = "text_padding";
constexpr const char *kSettingTextBgOpacity = "text_bg_opacity";
constexpr const char *kSettingLowConfidenceLabel = "low_conf_label";

constexpr uint64_t kOneSecondNs = 1'000'000'000ULL;
constexpr uint64_t kPerfLogIntervalNs = 5ULL * kOneSecondNs;

bool CopyPlaneToLinear(
  const uint8_t *src,
  const std::size_t src_stride,
  uint8_t *dst,
  const std::size_t row_bytes,
  const int rows)
{
  if (src == nullptr || dst == nullptr || row_bytes == 0 || rows <= 0 || src_stride < row_bytes) {
    return false;
  }

  for (int row = 0; row < rows; ++row) {
    std::memcpy(dst + static_cast<std::size_t>(row) * row_bytes, src + static_cast<std::size_t>(row) * src_stride, row_bytes);
  }
  return true;
}

bool CopyLinearToPlane(
  const uint8_t *src,
  uint8_t *dst,
  const std::size_t dst_stride,
  const std::size_t row_bytes,
  const int rows)
{
  if (src == nullptr || dst == nullptr || row_bytes == 0 || rows <= 0 || dst_stride < row_bytes) {
    return false;
  }

  for (int row = 0; row < rows; ++row) {
    std::memcpy(dst + static_cast<std::size_t>(row) * dst_stride, src + static_cast<std::size_t>(row) * row_bytes, row_bytes);
  }
  return true;
}

const char *VideoFormatToString(const video_format format)
{
  switch (format) {
  case VIDEO_FORMAT_I420:
    return "I420";
  case VIDEO_FORMAT_NV12:
    return "NV12";
  case VIDEO_FORMAT_YUY2:
    return "YUY2";
  case VIDEO_FORMAT_UYVY:
    return "UYVY";
  case VIDEO_FORMAT_RGBA:
    return "RGBA";
  case VIDEO_FORMAT_BGRA:
    return "BGRA";
  case VIDEO_FORMAT_BGRX:
    return "BGRX";
  case VIDEO_FORMAT_Y800:
    return "Y800";
  default:
    return "UNKNOWN";
  }
}

cv::Scalar MakeBgrColor(const int r, const int g, const int b, const int a = 255)
{
  return cv::Scalar(
    std::clamp(b, 0, 255),
    std::clamp(g, 0, 255),
    std::clamp(r, 0, 255),
    std::clamp(a, 0, 255));
}

cv::Scalar ResolveBoxColor(const FilterConfig &config, const Emotion emotion)
{
  if (config.box_use_emotion_color) {
    return EmotionColorBgra(emotion);
  }
  return MakeBgrColor(config.box_color_r, config.box_color_g, config.box_color_b);
}

cv::Scalar ResolveTextColor(const FilterConfig &config, const Emotion emotion)
{
  if (config.text_use_emotion_color) {
    return EmotionColorBgra(emotion);
  }
  return MakeBgrColor(config.text_color_r, config.text_color_g, config.text_color_b);
}

void FillRectWithOpacity(cv::Mat *image, const cv::Rect &rect, const cv::Scalar &color, const int opacity)
{
  if (image == nullptr || image->empty() || opacity <= 0) {
    return;
  }

  const cv::Rect frame_bounds(0, 0, image->cols, image->rows);
  const cv::Rect clipped = rect & frame_bounds;
  if (clipped.empty()) {
    return;
  }

  const int clamped_opacity = std::clamp(opacity, 0, 255);
  if (clamped_opacity >= 255) {
    cv::rectangle(*image, clipped, color, cv::FILLED, cv::LINE_AA);
    return;
  }

  cv::Mat roi = (*image)(clipped);
  cv::Mat overlay(roi.size(), roi.type(), color);
  const double alpha = static_cast<double>(clamped_opacity) / 255.0;
  cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
}

void DrawTextWithOpacity(
  cv::Mat *image,
  const std::string &text,
  const cv::Point &origin,
  const int font_face,
  const double font_scale,
  const cv::Scalar &color,
  const int thickness,
  const int line_type,
  const int opacity)
{
  if (image == nullptr || image->empty() || text.empty() || opacity <= 0) {
    return;
  }

  const int clamped_opacity = std::clamp(opacity, 0, 255);
  if (clamped_opacity >= 255) {
    cv::putText(*image, text, origin, font_face, font_scale, color, thickness, line_type);
    return;
  }

  int baseline = 0;
  const cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
  const cv::Rect text_rect(origin.x, origin.y - text_size.height, text_size.width + 2, text_size.height + baseline + 2);
  const cv::Rect frame_bounds(0, 0, image->cols, image->rows);
  const cv::Rect clipped = text_rect & frame_bounds;
  if (clipped.empty()) {
    return;
  }

  cv::Mat roi = (*image)(clipped);
  cv::Mat overlay = roi.clone();
  const cv::Point local_origin(origin.x - clipped.x, origin.y - clipped.y);
  cv::putText(overlay, text, local_origin, font_face, font_scale, color, thickness, line_type);

  const double alpha = static_cast<double>(clamped_opacity) / 255.0;
  cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
}

std::string BuildFaceText(const DetectedFace &face, const FilterConfig &config)
{
  const bool low_confidence = face.confidence < config.confidence_threshold;
  const char *base_label = low_confidence ? config.low_conf_label.c_str() : EmotionToFrenchLabel(face.label);

  std::ostringstream text_builder;
  if (config.show_track_id) {
    text_builder << "#" << face.track_id << " ";
  }
  text_builder << base_label;
  if (config.show_confidence) {
    text_builder << " " << static_cast<int>(std::round(face.confidence * 100.0f)) << "%";
  }
  return text_builder.str();
}

const char *FilterGetName(void *)
{
  return obs_module_text("FaceEmotionFilter.Name");
}

void *FilterCreate(obs_data_t *settings, obs_source_t *source)
{
  auto *filter = new FaceEmotionFilter(source);
  filter->Update(settings);
  return filter;
}

void FilterDestroy(void *data)
{
  delete static_cast<FaceEmotionFilter *>(data);
}

void FilterUpdate(void *data, obs_data_t *settings)
{
  static_cast<FaceEmotionFilter *>(data)->Update(settings);
}

obs_properties_t *FilterGetProperties(void *)
{
  return FaceEmotionFilter::GetProperties();
}

void FilterGetDefaults(obs_data_t *settings)
{
  FaceEmotionFilter::GetDefaults(settings);
}

void FilterTick(void *data, float seconds)
{
  static_cast<FaceEmotionFilter *>(data)->Tick(seconds);
}

obs_source_frame *FilterVideo(void *data, obs_source_frame *frame)
{
  return static_cast<FaceEmotionFilter *>(data)->FilterVideo(frame);
}

} // namespace

FaceEmotionFilter::FaceEmotionFilter(obs_source_t *source) : source_(source)
{
  const std::string face_model_path = ResolveModelPath(kFaceModelName);
  const std::string emotion_model_path = ResolveModelPath(kEmotionModelName);

  if (face_model_path.empty() || emotion_model_path.empty()) {
    obs_log(LOG_ERROR, "unable to resolve model paths; filter will stay disabled");
    return;
  }

  if (!std::filesystem::exists(face_model_path) || !std::filesystem::exists(emotion_model_path)) {
    obs_log(LOG_ERROR, "missing model file(s). expected under data/models/");
    return;
  }

  std::string error;
  if (!worker_.Start(face_model_path, emotion_model_path, BuildWorkerConfig(config_), &error)) {
    obs_log(LOG_ERROR, "failed to start inference worker: %s", error.c_str());
    return;
  }

  worker_ready_ = true;
}

FaceEmotionFilter::~FaceEmotionFilter()
{
  worker_.Stop();
}

void FaceEmotionFilter::Update(obs_data_t *settings)
{
  FilterConfig updated_config;
  updated_config.enabled = obs_data_get_bool(settings, kSettingEnabled);
  updated_config.max_faces = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingMaxFaces)), 1, 3);
  updated_config.inference_fps = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingInferenceFps)), 0, 240);
  updated_config.inference_width =
    std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingInferenceWidth)), 160, 1920);
  updated_config.confidence_threshold =
    std::clamp(static_cast<float>(obs_data_get_double(settings, kSettingConfidenceThreshold)), 0.0f, 1.0f);
  updated_config.smoothing_seconds =
    std::clamp(static_cast<float>(obs_data_get_double(settings, kSettingSmoothingSeconds)), 0.0f, 2.0f);
  updated_config.show_confidence = obs_data_get_bool(settings, kSettingShowConfidence);
  updated_config.show_box = obs_data_get_bool(settings, kSettingShowBox);
  updated_config.box_use_emotion_color = obs_data_get_bool(settings, kSettingBoxUseEmotionColor);
  updated_config.box_color_r = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingBoxColorR)), 0, 255);
  updated_config.box_color_g = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingBoxColorG)), 0, 255);
  updated_config.box_color_b = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingBoxColorB)), 0, 255);
  updated_config.box_thickness = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingBoxThickness)), 1, 12);
  updated_config.show_label = obs_data_get_bool(settings, kSettingShowLabel);
  updated_config.show_track_id = obs_data_get_bool(settings, kSettingShowTrackId);
  updated_config.text_use_emotion_color = obs_data_get_bool(settings, kSettingTextUseEmotionColor);
  updated_config.text_color_r = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingTextColorR)), 0, 255);
  updated_config.text_color_g = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingTextColorG)), 0, 255);
  updated_config.text_color_b = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingTextColorB)), 0, 255);
  updated_config.text_opacity = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingTextOpacity)), 0, 255);
  updated_config.text_scale =
    std::clamp(static_cast<float>(obs_data_get_double(settings, kSettingTextScale)), 0.4f, 3.0f);
  updated_config.text_thickness = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingTextThickness)), 1, 8);
  updated_config.text_padding = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingTextPadding)), 0, 20);
  updated_config.text_bg_opacity = std::clamp(static_cast<int>(obs_data_get_int(settings, kSettingTextBgOpacity)), 0, 255);

  const char *low_conf_label = obs_data_get_string(settings, kSettingLowConfidenceLabel);
  updated_config.low_conf_label = low_conf_label != nullptr && low_conf_label[0] != '\0' ? low_conf_label : "Incertain";

  {
    std::scoped_lock lock(config_mutex_);
    config_ = updated_config;
  }

  if (worker_ready_) {
    worker_.UpdateConfig(BuildWorkerConfig(updated_config));
  }
}

void FaceEmotionFilter::Tick(float)
{
}

obs_source_frame *FaceEmotionFilter::FilterVideo(obs_source_frame *frame)
{
  if (frame == nullptr || !worker_ready_) {
    return frame;
  }

  FilterConfig local_config;
  {
    std::scoped_lock lock(config_mutex_);
    local_config = config_;
  }

  if (!local_config.enabled) {
    return frame;
  }

  if (!SupportsFrameFormat(frame->format)) {
    if (!warned_unsupported_format_) {
      warned_unsupported_format_ = true;
      obs_log(
        LOG_WARNING,
        "unsupported frame format for %s: %s (%d)",
        kFilterId,
        VideoFormatToString(frame->format),
        static_cast<int>(frame->format));
    }
    return frame;
  }
  warned_unsupported_format_ = false;

  const uint64_t timestamp_ns = GetTimestampNs(frame);
  const int configured_fps = std::max(0, local_config.inference_fps);
  const bool unthrottled = configured_fps == 0;
  const uint64_t interval_ns = unthrottled ? 0 : (kOneSecondNs / static_cast<uint64_t>(configured_fps));

  cv::Mat current_bgr_frame;
  bool have_current_bgr = false;

  if (unthrottled || last_submitted_ts_ns_ == 0 || timestamp_ns >= last_submitted_ts_ns_ + interval_ns) {
    if (ExtractBgrFrame(frame, &current_bgr_frame)) {
      worker_.SubmitFrame(
        current_bgr_frame,
        timestamp_ns,
        static_cast<int>(frame->width),
        static_cast<int>(frame->height));
      if (!unthrottled) {
        last_submitted_ts_ns_ = timestamp_ns;
      }
      have_current_bgr = true;
    }
  }

  std::vector<DetectedFace> faces;
  double inference_ms = 0.0;
  if (worker_.TryConsumeLatest(&faces, &inference_ms, nullptr)) {
    latest_faces_ = std::move(faces);
    perf_total_ms_ += inference_ms;
    perf_samples_++;
    perf_results_++;
  }

  if (local_config.show_box || local_config.show_label) {
    if (SupportsInPlaceOverlay(frame->format)) {
      DrawOverlay(frame, latest_faces_, local_config);
    } else {
      if (!have_current_bgr) {
        have_current_bgr = ExtractBgrFrame(frame, &current_bgr_frame);
      }

      if (have_current_bgr) {
        DrawOverlayOnBgr(&current_bgr_frame, latest_faces_, local_config);
        WriteBgrFrame(frame, current_bgr_frame);
      }
    }
  }

  LogPerfEveryFiveSeconds(timestamp_ns);
  return frame;
}

void FaceEmotionFilter::GetDefaults(obs_data_t *settings)
{
  obs_data_set_default_bool(settings, kSettingEnabled, true);
  obs_data_set_default_int(settings, kSettingMaxFaces, 3);
  obs_data_set_default_int(settings, kSettingInferenceFps, 0);
  obs_data_set_default_int(settings, kSettingInferenceWidth, 640);
  obs_data_set_default_double(settings, kSettingConfidenceThreshold, 0.30);
  obs_data_set_default_double(settings, kSettingSmoothingSeconds, 0.6);
  obs_data_set_default_bool(settings, kSettingShowConfidence, true);
  obs_data_set_default_bool(settings, kSettingShowBox, true);
  obs_data_set_default_bool(settings, kSettingBoxUseEmotionColor, true);
  obs_data_set_default_int(settings, kSettingBoxColorR, 0);
  obs_data_set_default_int(settings, kSettingBoxColorG, 255);
  obs_data_set_default_int(settings, kSettingBoxColorB, 0);
  obs_data_set_default_int(settings, kSettingBoxThickness, 2);
  obs_data_set_default_bool(settings, kSettingShowLabel, true);
  obs_data_set_default_bool(settings, kSettingShowTrackId, false);
  obs_data_set_default_bool(settings, kSettingTextUseEmotionColor, true);
  obs_data_set_default_int(settings, kSettingTextColorR, 255);
  obs_data_set_default_int(settings, kSettingTextColorG, 255);
  obs_data_set_default_int(settings, kSettingTextColorB, 255);
  obs_data_set_default_int(settings, kSettingTextOpacity, 255);
  obs_data_set_default_double(settings, kSettingTextScale, 1.15);
  obs_data_set_default_int(settings, kSettingTextThickness, 2);
  obs_data_set_default_int(settings, kSettingTextPadding, 4);
  obs_data_set_default_int(settings, kSettingTextBgOpacity, 0);
  obs_data_set_default_string(settings, kSettingLowConfidenceLabel, "Incertain");
}

obs_properties_t *FaceEmotionFilter::GetProperties()
{
  obs_properties_t *props = obs_properties_create();
  obs_properties_add_bool(props, kSettingEnabled, obs_module_text("FaceEmotionFilter.Enabled"));
  obs_properties_add_int_slider(props, kSettingMaxFaces, obs_module_text("FaceEmotionFilter.MaxFaces"), 1, 3, 1);
  obs_properties_add_int_slider(props, kSettingInferenceFps, obs_module_text("FaceEmotionFilter.InferenceFps"), 0, 240, 1);
  obs_properties_add_int_slider(
    props,
    kSettingInferenceWidth,
    obs_module_text("FaceEmotionFilter.InferenceWidth"),
    320,
    1280,
    32);
  obs_properties_add_float_slider(
    props,
    kSettingConfidenceThreshold,
    obs_module_text("FaceEmotionFilter.ConfidenceThreshold"),
    0.10,
    0.90,
    0.01);
  obs_properties_add_float_slider(
    props,
    kSettingSmoothingSeconds,
    obs_module_text("FaceEmotionFilter.SmoothingSeconds"),
    0.0,
    2.0,
    0.1);
  obs_properties_add_bool(props, kSettingShowBox, obs_module_text("FaceEmotionFilter.ShowBox"));
  obs_properties_add_bool(
    props,
    kSettingBoxUseEmotionColor,
    obs_module_text("FaceEmotionFilter.BoxUseEmotionColor"));
  obs_properties_add_int_slider(props, kSettingBoxColorR, obs_module_text("FaceEmotionFilter.BoxColorR"), 0, 255, 1);
  obs_properties_add_int_slider(props, kSettingBoxColorG, obs_module_text("FaceEmotionFilter.BoxColorG"), 0, 255, 1);
  obs_properties_add_int_slider(props, kSettingBoxColorB, obs_module_text("FaceEmotionFilter.BoxColorB"), 0, 255, 1);
  obs_properties_add_int_slider(
    props,
    kSettingBoxThickness,
    obs_module_text("FaceEmotionFilter.BoxThickness"),
    1,
    12,
    1);
  obs_properties_add_bool(props, kSettingShowLabel, obs_module_text("FaceEmotionFilter.ShowLabel"));
  obs_properties_add_bool(props, kSettingShowTrackId, obs_module_text("FaceEmotionFilter.ShowTrackId"));
  obs_properties_add_bool(
    props,
    kSettingTextUseEmotionColor,
    obs_module_text("FaceEmotionFilter.TextUseEmotionColor"));
  obs_properties_add_int_slider(props, kSettingTextColorR, obs_module_text("FaceEmotionFilter.TextColorR"), 0, 255, 1);
  obs_properties_add_int_slider(props, kSettingTextColorG, obs_module_text("FaceEmotionFilter.TextColorG"), 0, 255, 1);
  obs_properties_add_int_slider(props, kSettingTextColorB, obs_module_text("FaceEmotionFilter.TextColorB"), 0, 255, 1);
  obs_properties_add_int_slider(props, kSettingTextOpacity, obs_module_text("FaceEmotionFilter.TextOpacity"), 0, 255, 1);
  obs_properties_add_float_slider(props, kSettingTextScale, obs_module_text("FaceEmotionFilter.TextScale"), 0.4, 3.0, 0.05);
  obs_properties_add_int_slider(
    props,
    kSettingTextThickness,
    obs_module_text("FaceEmotionFilter.TextThickness"),
    1,
    8,
    1);
  obs_properties_add_int_slider(props, kSettingTextPadding, obs_module_text("FaceEmotionFilter.TextPadding"), 0, 20, 1);
  obs_properties_add_int_slider(
    props,
    kSettingTextBgOpacity,
    obs_module_text("FaceEmotionFilter.TextBgOpacity"),
    0,
    255,
    1);
  obs_properties_add_bool(props, kSettingShowConfidence, obs_module_text("FaceEmotionFilter.ShowConfidence"));
  obs_properties_add_text(
    props,
    kSettingLowConfidenceLabel,
    obs_module_text("FaceEmotionFilter.LowConfidenceLabel"),
    OBS_TEXT_DEFAULT);
  return props;
}

uint64_t FaceEmotionFilter::GetTimestampNs(const obs_source_frame *frame)
{
  if (frame != nullptr && frame->timestamp > 0) {
    return frame->timestamp;
  }

  return os_gettime_ns();
}

bool FaceEmotionFilter::ExtractBgrFrame(const obs_source_frame *frame, cv::Mat *bgr_frame) const
{
  if (frame == nullptr || bgr_frame == nullptr || frame->data[0] == nullptr || frame->width == 0 || frame->height == 0) {
    return false;
  }

  const int width = static_cast<int>(frame->width);
  const int height = static_cast<int>(frame->height);

  switch (frame->format) {
  case VIDEO_FORMAT_BGRA:
  case VIDEO_FORMAT_BGRX:
  case VIDEO_FORMAT_RGBA:
  {
    cv::Mat rgba_mat(height, width, CV_8UC4, frame->data[0], static_cast<std::size_t>(frame->linesize[0]));
    const int code = (frame->format == VIDEO_FORMAT_BGRA || frame->format == VIDEO_FORMAT_BGRX)
      ? cv::COLOR_BGRA2BGR
      : cv::COLOR_RGBA2BGR;
    cv::cvtColor(rgba_mat, *bgr_frame, code);
    return true;
  }
  case VIDEO_FORMAT_NV12: {
    if (frame->data[1] == nullptr) {
      return false;
    }
    cv::Mat y_plane(height, width, CV_8UC1, frame->data[0], static_cast<std::size_t>(frame->linesize[0]));
    cv::Mat uv_plane(height / 2, width / 2, CV_8UC2, frame->data[1], static_cast<std::size_t>(frame->linesize[1]));
    cv::cvtColorTwoPlane(y_plane, uv_plane, *bgr_frame, cv::COLOR_YUV2BGR_NV12);
    return true;
  }
  case VIDEO_FORMAT_I420: {
    if (frame->data[1] == nullptr || frame->data[2] == nullptr) {
      return false;
    }

    cv::Mat i420(height * 3 / 2, width, CV_8UC1);
    uint8_t *dst = i420.data;
    if (!CopyPlaneToLinear(
          frame->data[0],
          static_cast<std::size_t>(frame->linesize[0]),
          dst,
          static_cast<std::size_t>(width),
          height)) {
      return false;
    }

    const int chroma_width = width / 2;
    const int chroma_height = height / 2;
    uint8_t *u_dst = dst + static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    uint8_t *v_dst = u_dst + static_cast<std::size_t>(chroma_width) * static_cast<std::size_t>(chroma_height);

    if (!CopyPlaneToLinear(
          frame->data[1],
          static_cast<std::size_t>(frame->linesize[1]),
          u_dst,
          static_cast<std::size_t>(chroma_width),
          chroma_height)) {
      return false;
    }

    if (!CopyPlaneToLinear(
          frame->data[2],
          static_cast<std::size_t>(frame->linesize[2]),
          v_dst,
          static_cast<std::size_t>(chroma_width),
          chroma_height)) {
      return false;
    }

    cv::cvtColor(i420, *bgr_frame, cv::COLOR_YUV2BGR_I420);
    return true;
  }
  case VIDEO_FORMAT_YUY2: {
    cv::Mat yuy2(height, width, CV_8UC2, frame->data[0], static_cast<std::size_t>(frame->linesize[0]));
    cv::cvtColor(yuy2, *bgr_frame, cv::COLOR_YUV2BGR_YUY2);
    return true;
  }
  case VIDEO_FORMAT_UYVY: {
    cv::Mat uyvy(height, width, CV_8UC2, frame->data[0], static_cast<std::size_t>(frame->linesize[0]));
    cv::cvtColor(uyvy, *bgr_frame, cv::COLOR_YUV2BGR_UYVY);
    return true;
  }
  case VIDEO_FORMAT_Y800: {
    cv::Mat gray(height, width, CV_8UC1, frame->data[0], static_cast<std::size_t>(frame->linesize[0]));
    cv::cvtColor(gray, *bgr_frame, cv::COLOR_GRAY2BGR);
    return true;
  }
  default:
    return false;
  }
}

bool FaceEmotionFilter::WriteBgrFrame(obs_source_frame *frame, const cv::Mat &bgr_frame) const
{
  if (frame == nullptr || frame->data[0] == nullptr || bgr_frame.empty()) {
    return false;
  }

  const int width = static_cast<int>(frame->width);
  const int height = static_cast<int>(frame->height);
  if (bgr_frame.cols != width || bgr_frame.rows != height || bgr_frame.type() != CV_8UC3) {
    return false;
  }

  switch (frame->format) {
  case VIDEO_FORMAT_BGRA:
  case VIDEO_FORMAT_BGRX:
  case VIDEO_FORMAT_RGBA:
  {
    cv::Mat dst_rgba(height, width, CV_8UC4, frame->data[0], static_cast<std::size_t>(frame->linesize[0]));
    const int code = (frame->format == VIDEO_FORMAT_BGRA || frame->format == VIDEO_FORMAT_BGRX)
      ? cv::COLOR_BGR2BGRA
      : cv::COLOR_BGR2RGBA;
    cv::cvtColor(bgr_frame, dst_rgba, code);
    return true;
  }
  case VIDEO_FORMAT_Y800: {
    cv::Mat gray;
    cv::cvtColor(bgr_frame, gray, cv::COLOR_BGR2GRAY);
    return CopyLinearToPlane(
      gray.data,
      frame->data[0],
      static_cast<std::size_t>(frame->linesize[0]),
      static_cast<std::size_t>(width),
      height);
  }
  case VIDEO_FORMAT_YUY2: {
    cv::Mat yuy2;
    cv::cvtColor(bgr_frame, yuy2, cv::COLOR_BGR2YUV_YUY2);
    return CopyLinearToPlane(
      yuy2.data,
      frame->data[0],
      static_cast<std::size_t>(frame->linesize[0]),
      static_cast<std::size_t>(width) * 2ULL,
      height);
  }
  case VIDEO_FORMAT_UYVY: {
    cv::Mat uyvy;
    cv::cvtColor(bgr_frame, uyvy, cv::COLOR_BGR2YUV_UYVY);
    return CopyLinearToPlane(
      uyvy.data,
      frame->data[0],
      static_cast<std::size_t>(frame->linesize[0]),
      static_cast<std::size_t>(width) * 2ULL,
      height);
  }
  case VIDEO_FORMAT_I420:
  case VIDEO_FORMAT_NV12: {
    cv::Mat i420;
    cv::cvtColor(bgr_frame, i420, cv::COLOR_BGR2YUV_I420);

    const std::size_t y_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const int chroma_width = width / 2;
    const int chroma_height = height / 2;
    const std::size_t chroma_size = static_cast<std::size_t>(chroma_width) * static_cast<std::size_t>(chroma_height);

    const uint8_t *src_y = i420.data;
    const uint8_t *src_u = src_y + y_size;
    const uint8_t *src_v = src_u + chroma_size;

    if (!CopyLinearToPlane(
          src_y,
          frame->data[0],
          static_cast<std::size_t>(frame->linesize[0]),
          static_cast<std::size_t>(width),
          height)) {
      return false;
    }

    if (frame->format == VIDEO_FORMAT_I420) {
      if (frame->data[1] == nullptr || frame->data[2] == nullptr) {
        return false;
      }

      if (!CopyLinearToPlane(
            src_u,
            frame->data[1],
            static_cast<std::size_t>(frame->linesize[1]),
            static_cast<std::size_t>(chroma_width),
            chroma_height)) {
        return false;
      }

      return CopyLinearToPlane(
        src_v,
        frame->data[2],
        static_cast<std::size_t>(frame->linesize[2]),
        static_cast<std::size_t>(chroma_width),
        chroma_height);
    }

    if (frame->data[1] == nullptr || frame->linesize[1] < static_cast<uint32_t>(width)) {
      return false;
    }

    for (int row = 0; row < chroma_height; ++row) {
      uint8_t *dst_uv = frame->data[1] + static_cast<std::size_t>(row) * static_cast<std::size_t>(frame->linesize[1]);
      const uint8_t *row_u = src_u + static_cast<std::size_t>(row) * static_cast<std::size_t>(chroma_width);
      const uint8_t *row_v = src_v + static_cast<std::size_t>(row) * static_cast<std::size_t>(chroma_width);
      for (int col = 0; col < chroma_width; ++col) {
        dst_uv[2 * col] = row_u[col];
        dst_uv[2 * col + 1] = row_v[col];
      }
    }
    return true;
  }
  default:
    return false;
  }
}

bool FaceEmotionFilter::SupportsFrameFormat(const video_format format)
{
  switch (format) {
  case VIDEO_FORMAT_BGRA:
  case VIDEO_FORMAT_BGRX:
  case VIDEO_FORMAT_RGBA:
  case VIDEO_FORMAT_NV12:
  case VIDEO_FORMAT_I420:
  case VIDEO_FORMAT_YUY2:
  case VIDEO_FORMAT_UYVY:
  case VIDEO_FORMAT_Y800:
    return true;
  default:
    return false;
  }
}

bool FaceEmotionFilter::SupportsInPlaceOverlay(const video_format format)
{
  switch (format) {
  case VIDEO_FORMAT_BGRA:
  case VIDEO_FORMAT_BGRX:
    return true;
  default:
    return false;
  }
}

void FaceEmotionFilter::DrawOverlay(
  obs_source_frame *frame,
  const std::vector<DetectedFace> &faces,
  const FilterConfig &config)
{
  if (frame == nullptr || frame->data[0] == nullptr || (!config.show_box && !config.show_label)) {
    return;
  }

  cv::Mat draw_frame(
    static_cast<int>(frame->height),
    static_cast<int>(frame->width),
    CV_8UC4,
    frame->data[0],
    static_cast<std::size_t>(frame->linesize[0]));
  const cv::Rect bounds(0, 0, draw_frame.cols, draw_frame.rows);

  for (const auto &face : faces) {
    cv::Rect bbox = face.bbox & bounds;
    if (bbox.empty()) {
      continue;
    }

    const cv::Scalar box_color = ResolveBoxColor(config, face.label);
    const cv::Scalar text_color = ResolveTextColor(config, face.label);

    if (config.show_box) {
      cv::rectangle(draw_frame, bbox, box_color, config.box_thickness, cv::LINE_AA);
    }

    if (config.show_label) {
      const std::string text = BuildFaceText(face, config);
      const int safe_padding = std::max(0, config.text_padding);
      const int safe_text_thickness = std::max(1, config.text_thickness);
      const double safe_text_scale = std::max(0.1, static_cast<double>(config.text_scale));

      int baseline = 0;
      const cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, safe_text_scale, safe_text_thickness, &baseline);

      int text_x = bbox.x;
      int text_y = bbox.y - (safe_padding + 4);
      if (text_y < text_size.height + safe_padding) {
        text_y = bbox.y + text_size.height + safe_padding + 4;
      }
      if (text_x + text_size.width + (2 * safe_padding) > draw_frame.cols) {
        text_x = std::max(0, draw_frame.cols - text_size.width - (2 * safe_padding));
      }

      const int rect_x = std::max(0, text_x - safe_padding);
      const int rect_y = std::max(0, text_y - text_size.height - safe_padding);
      const cv::Rect label_rect(
        rect_x,
        rect_y,
        std::max(0, std::min(text_size.width + (2 * safe_padding), draw_frame.cols - rect_x)),
        std::max(0, std::min(text_size.height + (2 * safe_padding), draw_frame.rows - rect_y)));

      FillRectWithOpacity(&draw_frame, label_rect, cv::Scalar(0, 0, 0, 255), config.text_bg_opacity);
      DrawTextWithOpacity(
        &draw_frame,
        text,
        cv::Point(text_x, text_y),
        cv::FONT_HERSHEY_SIMPLEX,
        safe_text_scale,
        text_color,
        safe_text_thickness,
        cv::LINE_AA,
        config.text_opacity);
    }
  }
}

void FaceEmotionFilter::DrawOverlayOnBgr(
  cv::Mat *bgr_frame,
  const std::vector<DetectedFace> &faces,
  const FilterConfig &config) const
{
  if (bgr_frame == nullptr || bgr_frame->empty() || bgr_frame->type() != CV_8UC3 || (!config.show_box && !config.show_label)) {
    return;
  }

  const cv::Rect bounds(0, 0, bgr_frame->cols, bgr_frame->rows);

  for (const auto &face : faces) {
    cv::Rect bbox = face.bbox & bounds;
    if (bbox.empty()) {
      continue;
    }

    const cv::Scalar box_color = ResolveBoxColor(config, face.label);
    const cv::Scalar text_color = ResolveTextColor(config, face.label);

    if (config.show_box) {
      cv::rectangle(*bgr_frame, bbox, box_color, config.box_thickness, cv::LINE_AA);
    }

    if (config.show_label) {
      const std::string text = BuildFaceText(face, config);
      const int safe_padding = std::max(0, config.text_padding);
      const int safe_text_thickness = std::max(1, config.text_thickness);
      const double safe_text_scale = std::max(0.1, static_cast<double>(config.text_scale));

      int baseline = 0;
      const cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, safe_text_scale, safe_text_thickness, &baseline);

      int text_x = bbox.x;
      int text_y = bbox.y - (safe_padding + 4);
      if (text_y < text_size.height + safe_padding) {
        text_y = bbox.y + text_size.height + safe_padding + 4;
      }
      if (text_x + text_size.width + (2 * safe_padding) > bgr_frame->cols) {
        text_x = std::max(0, bgr_frame->cols - text_size.width - (2 * safe_padding));
      }

      const int rect_x = std::max(0, text_x - safe_padding);
      const int rect_y = std::max(0, text_y - text_size.height - safe_padding);
      const cv::Rect label_rect(
        rect_x,
        rect_y,
        std::max(0, std::min(text_size.width + (2 * safe_padding), bgr_frame->cols - rect_x)),
        std::max(0, std::min(text_size.height + (2 * safe_padding), bgr_frame->rows - rect_y)));

      FillRectWithOpacity(bgr_frame, label_rect, cv::Scalar(0, 0, 0), config.text_bg_opacity);
      DrawTextWithOpacity(
        bgr_frame,
        text,
        cv::Point(text_x, text_y),
        cv::FONT_HERSHEY_SIMPLEX,
        safe_text_scale,
        text_color,
        safe_text_thickness,
        cv::LINE_AA,
        config.text_opacity);
    }
  }
}

void FaceEmotionFilter::LogPerfEveryFiveSeconds(const uint64_t now_ns)
{
  if (perf_window_start_ns_ == 0) {
    perf_window_start_ns_ = now_ns;
    return;
  }

  const uint64_t elapsed_ns = now_ns - perf_window_start_ns_;
  if (elapsed_ns < kPerfLogIntervalNs) {
    return;
  }

  const double elapsed_seconds = static_cast<double>(elapsed_ns) / static_cast<double>(kOneSecondNs);
  const double avg_inference_ms = perf_samples_ > 0 ? perf_total_ms_ / static_cast<double>(perf_samples_) : 0.0;
  const double inference_fps = elapsed_seconds > 0.0 ? static_cast<double>(perf_results_) / elapsed_seconds : 0.0;
  const char *top_label = "none";
  float top_conf = 0.0f;
  if (!latest_faces_.empty()) {
    top_label = EmotionToFrenchLabel(latest_faces_.front().label);
    top_conf = latest_faces_.front().confidence;
  }

  obs_log(
    LOG_INFO,
    "perf avg_inference_ms=%.2f inference_fps=%.2f queue=%zu top_label=%s top_conf=%.2f",
    avg_inference_ms,
    inference_fps,
    worker_.QueueSize(),
    top_label,
    top_conf);

  perf_window_start_ns_ = now_ns;
  perf_total_ms_ = 0.0;
  perf_samples_ = 0;
  perf_results_ = 0;
}

std::string FaceEmotionFilter::ResolveModelPath(const char *model_name) const
{
  const char *module_data_path = obs_get_module_data_path(obs_current_module());
  if (module_data_path == nullptr || model_name == nullptr) {
    return {};
  }

  const std::filesystem::path path = std::filesystem::path(module_data_path) / "models" / model_name;
  return path.string();
}

InferenceWorker::Config FaceEmotionFilter::BuildWorkerConfig(const FilterConfig &filter_config)
{
  InferenceWorker::Config config;
  config.max_faces = filter_config.max_faces;
  config.inference_width = filter_config.inference_width;
  config.confidence_threshold = filter_config.confidence_threshold;
  config.smoothing_seconds = filter_config.smoothing_seconds;
  return config;
}

const obs_source_info *GetFaceEmotionFilterSourceInfo()
{
  static obs_source_info info = {};
  static bool initialized = false;
  if (!initialized) {
    info.id = kFilterId;
    info.type = OBS_SOURCE_TYPE_FILTER;
    info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC_VIDEO;
    info.get_name = FilterGetName;
    info.create = FilterCreate;
    info.destroy = FilterDestroy;
    info.update = FilterUpdate;
    info.get_defaults = FilterGetDefaults;
    info.get_properties = FilterGetProperties;
    info.video_tick = FilterTick;
    info.filter_video = FilterVideo;
    initialized = true;
  }

  return &info;
}
