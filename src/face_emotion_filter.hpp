#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <obs-module.h>

#include "inference_worker.hpp"

struct FilterConfig {
  bool enabled = true;
  int max_faces = 3;
  int inference_fps = 15;
  int inference_width = 640;
  float confidence_threshold = 0.30f;
  float smoothing_seconds = 0.6f;
  bool show_confidence = true;
  bool show_box = true;
  bool box_use_emotion_color = true;
  int box_color_r = 0;
  int box_color_g = 255;
  int box_color_b = 0;
  int box_thickness = 2;
  bool show_label = true;
  bool show_track_id = false;
  bool text_use_emotion_color = true;
  int text_color_r = 255;
  int text_color_g = 255;
  int text_color_b = 255;
  int text_opacity = 255;
  float text_scale = 1.15f;
  int text_thickness = 2;
  int text_padding = 4;
  int text_bg_opacity = 180;
  std::string low_conf_label = "Incertain";
};

class FaceEmotionFilter {
public:
  explicit FaceEmotionFilter(obs_source_t *source);
  ~FaceEmotionFilter();

  void Update(obs_data_t *settings);
  void Tick(float seconds);
  obs_source_frame *FilterVideo(obs_source_frame *frame);

  static void GetDefaults(obs_data_t *settings);
  static obs_properties_t *GetProperties();

private:
  static uint64_t GetTimestampNs(const obs_source_frame *frame);
  bool ExtractBgrFrame(const obs_source_frame *frame, cv::Mat *bgr_frame) const;
  bool WriteBgrFrame(obs_source_frame *frame, const cv::Mat &bgr_frame) const;
  static bool SupportsFrameFormat(video_format format);
  static bool SupportsInPlaceOverlay(video_format format);
  void DrawOverlay(obs_source_frame *frame, const std::vector<DetectedFace> &faces, const FilterConfig &config);
  void DrawOverlayOnBgr(cv::Mat *bgr_frame, const std::vector<DetectedFace> &faces, const FilterConfig &config) const;
  void LogPerfEveryFiveSeconds(uint64_t now_ns);
  std::string ResolveModelPath(const char *model_name) const;
  static InferenceWorker::Config BuildWorkerConfig(const FilterConfig &filter_config);

  obs_source_t *source_ = nullptr;
  FilterConfig config_;
  mutable std::mutex config_mutex_;
  InferenceWorker worker_;
  bool worker_ready_ = false;
  bool warned_unsupported_format_ = false;
  uint64_t last_submitted_ts_ns_ = 0;

  std::vector<DetectedFace> latest_faces_;

  uint64_t perf_window_start_ns_ = 0;
  double perf_total_ms_ = 0.0;
  uint64_t perf_samples_ = 0;
  uint64_t perf_results_ = 0;
};

const obs_source_info *GetFaceEmotionFilterSourceInfo();
