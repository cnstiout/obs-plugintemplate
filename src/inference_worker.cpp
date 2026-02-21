#include "inference_worker.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace {

// Keep only the latest frame to minimize overlay latency under load.
constexpr std::size_t kMaxQueueSize = 1;
constexpr int kEmotionInputSize = 64;

cv::Rect ClampRectToFrame(const cv::Rect &rect, const int width, const int height)
{
  const cv::Rect frame_rect(0, 0, width, height);
  return rect & frame_rect;
}

cv::Rect MakeSquareRect(const cv::Rect &rect, const int frame_width, const int frame_height)
{
  if (rect.empty()) {
    return rect;
  }

  const int side = std::max(rect.width, rect.height);
  const int cx = rect.x + rect.width / 2;
  const int cy = rect.y + rect.height / 2;

  cv::Rect square(cx - side / 2, cy - side / 2, side, side);
  square = ClampRectToFrame(square, frame_width, frame_height);

  if (square.width <= 1 || square.height <= 1) {
    return ClampRectToFrame(rect, frame_width, frame_height);
  }
  return square;
}

} // namespace

InferenceWorker::InferenceWorker() = default;

InferenceWorker::~InferenceWorker()
{
  Stop();
}

bool InferenceWorker::Start(
  const std::string &face_model_path,
  const std::string &emotion_model_path,
  const Config &config,
  std::string *error)
{
  Stop();

  try {
    face_detector_ = cv::FaceDetectorYN::create(face_model_path, "", cv::Size(320, 320), 0.7f, 0.3f, 5000);
    emotion_net_ = cv::dnn::readNetFromONNX(emotion_model_path);
    if (face_detector_.empty()) {
      if (error != nullptr) {
        *error = "Face detector initialization failed.";
      }
      return false;
    }
    if (emotion_net_.empty()) {
      if (error != nullptr) {
        *error = "Emotion model initialization failed.";
      }
      return false;
    }
  } catch (const std::exception &ex) {
    if (error != nullptr) {
      *error = ex.what();
    }
    return false;
  }

  {
    std::scoped_lock lock(config_mutex_);
    config_ = config;
  }
  {
    std::scoped_lock queue_lock(queue_mutex_);
    queue_.clear();
    stop_requested_ = false;
  }
  {
    std::scoped_lock result_lock(result_mutex_);
    latest_result_ = {};
  }
  tracker_.Reset();

  worker_thread_ = std::thread(&InferenceWorker::WorkerLoop, this);
  running_ = true;
  return true;
}

void InferenceWorker::Stop()
{
  {
    std::scoped_lock lock(queue_mutex_);
    stop_requested_ = true;
  }
  queue_cv_.notify_all();

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }

  {
    std::scoped_lock lock(queue_mutex_);
    queue_.clear();
    stop_requested_ = false;
  }
  {
    std::scoped_lock result_lock(result_mutex_);
    latest_result_ = {};
  }
  running_ = false;
  tracker_.Reset();
}

void InferenceWorker::UpdateConfig(const Config &config)
{
  std::scoped_lock lock(config_mutex_);
  config_ = config;
}

void InferenceWorker::SubmitFrame(const cv::Mat &bgr_frame, const uint64_t timestamp_ns, const int source_width, const int source_height)
{
  if (!running_ || bgr_frame.empty()) {
    return;
  }

  FrameTask task;
  task.bgr_frame = bgr_frame.clone();
  task.timestamp_ns = timestamp_ns;
  task.source_width = source_width;
  task.source_height = source_height;

  {
    std::scoped_lock lock(queue_mutex_);
    if (queue_.size() >= kMaxQueueSize) {
      queue_.pop_front();
    }
    queue_.push_back(std::move(task));
  }

  queue_cv_.notify_one();
}

bool InferenceWorker::TryConsumeLatest(std::vector<DetectedFace> *faces, double *inference_ms, uint64_t *timestamp_ns)
{
  std::scoped_lock lock(result_mutex_);
  if (!latest_result_.valid) {
    return false;
  }

  if (faces != nullptr) {
    *faces = latest_result_.faces;
  }
  if (inference_ms != nullptr) {
    *inference_ms = latest_result_.inference_ms;
  }
  if (timestamp_ns != nullptr) {
    *timestamp_ns = latest_result_.timestamp_ns;
  }

  latest_result_.valid = false;
  return true;
}

std::size_t InferenceWorker::QueueSize() const
{
  std::scoped_lock lock(queue_mutex_);
  return queue_.size();
}

bool InferenceWorker::IsRunning() const
{
  return running_.load();
}

void InferenceWorker::WorkerLoop()
{
  for (;;) {
    FrameTask task;
    {
      std::unique_lock lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return stop_requested_ || !queue_.empty(); });

      if (stop_requested_ && queue_.empty()) {
        break;
      }

      task = std::move(queue_.front());
      queue_.pop_front();
    }

    const auto started = std::chrono::steady_clock::now();
    std::vector<DetectedFace> detections = RunInference(task);
    const auto ended = std::chrono::steady_clock::now();
    const double inference_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(ended - started).count();

    ResultPacket result;
    result.faces = std::move(detections);
    result.inference_ms = inference_ms;
    result.timestamp_ns = task.timestamp_ns;
    result.valid = true;

    {
      std::scoped_lock lock(result_mutex_);
      latest_result_ = std::move(result);
    }
  }
}

std::vector<DetectedFace> InferenceWorker::RunInference(const FrameTask &task)
{
  Config config;
  {
    std::scoped_lock lock(config_mutex_);
    config = config_;
  }

  if (task.bgr_frame.empty()) {
    tracker_.Reset();
    return {};
  }

  cv::Mat inference_frame = task.bgr_frame;
  double scale = 1.0;
  if (config.inference_width > 0 && task.bgr_frame.cols > config.inference_width) {
    scale = static_cast<double>(config.inference_width) / static_cast<double>(task.bgr_frame.cols);
    const int resized_height = std::max(1, static_cast<int>(std::round(task.bgr_frame.rows * scale)));
    cv::resize(task.bgr_frame, inference_frame, cv::Size(config.inference_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
  }

  std::vector<RawFaceDetection> detections;
  try {
    face_detector_->setInputSize(inference_frame.size());
    cv::Mat face_matrix;
    const int face_count = face_detector_->detect(inference_frame, face_matrix);
    if (face_count > 0 && !face_matrix.empty()) {
      for (int row = 0; row < face_matrix.rows; ++row) {
        if (face_matrix.cols < 15) {
          continue;
        }

        const float score = face_matrix.at<float>(row, 14);
        if (score <= 0.0f) {
          continue;
        }

        const float x = face_matrix.at<float>(row, 0);
        const float y = face_matrix.at<float>(row, 1);
        const float width = face_matrix.at<float>(row, 2);
        const float height = face_matrix.at<float>(row, 3);

        const int mapped_x = static_cast<int>(std::round(x / scale));
        const int mapped_y = static_cast<int>(std::round(y / scale));
        const int mapped_w = static_cast<int>(std::round(width / scale));
        const int mapped_h = static_cast<int>(std::round(height / scale));
        cv::Rect bbox(mapped_x, mapped_y, mapped_w, mapped_h);
        bbox = ClampRectToFrame(bbox, task.source_width, task.source_height);
        if (bbox.empty()) {
          continue;
        }

        const cv::Rect emotion_roi = MakeSquareRect(bbox, task.source_width, task.source_height);
        const auto probabilities = InferEmotion(task.bgr_frame(emotion_roi));
        detections.push_back(RawFaceDetection {bbox, probabilities});
      }
    }
  } catch (...) {
    tracker_.Reset();
    return {};
  }

  return tracker_.Update(
    detections,
    task.timestamp_ns,
    config.max_faces,
    config.smoothing_seconds,
    config.confidence_threshold);
}

std::array<float, kEmotionClassCount> InferenceWorker::InferEmotion(const cv::Mat &face_bgr)
{
  std::array<float, kEmotionClassCount> model_output {};
  if (face_bgr.empty()) {
    return model_output;
  }

  cv::Mat gray_face;
  cv::cvtColor(face_bgr, gray_face, cv::COLOR_BGR2GRAY);
  cv::resize(gray_face, gray_face, cv::Size(kEmotionInputSize, kEmotionInputSize), 0.0, 0.0, cv::INTER_LINEAR);
  cv::equalizeHist(gray_face, gray_face);

  cv::Mat gray_float;
  gray_face.convertTo(gray_float, CV_32F);
  const cv::Mat blob = cv::dnn::blobFromImage(
    gray_float,
    1.0,
    cv::Size(kEmotionInputSize, kEmotionInputSize),
    cv::Scalar(),
    false,
    false,
    CV_32F);

  emotion_net_.setInput(blob);
  const cv::Mat output = emotion_net_.forward();
  if (output.empty()) {
    return model_output;
  }

  const cv::Mat flattened = output.reshape(1, 1);
  const int count = std::min<int>(static_cast<int>(kEmotionClassCount), flattened.cols);
  for (int i = 0; i < count; ++i) {
    model_output[static_cast<std::size_t>(i)] = flattened.at<float>(0, i);
  }

  return NormalizeEmotionOutput(model_output);
}

bool InferenceWorker::LooksLikeProbabilities(const std::array<float, kEmotionClassCount> &values)
{
  float sum = 0.0f;
  for (const float value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
    if (value < -0.001f || value > 1.001f) {
      return false;
    }
    sum += value;
  }
  return sum > 0.85f && sum < 1.15f;
}

std::array<float, kEmotionClassCount> InferenceWorker::NormalizeEmotionOutput(
  const std::array<float, kEmotionClassCount> &model_output)
{
  std::array<float, kEmotionClassCount> probs {};

  if (LooksLikeProbabilities(model_output)) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < kEmotionClassCount; ++i) {
      probs[i] = std::clamp(model_output[i], 0.0f, 1.0f);
      sum += probs[i];
    }

    if (sum > std::numeric_limits<float>::epsilon()) {
      for (float &value : probs) {
        value /= sum;
      }
      return probs;
    }
  }

  const float max_logit = *std::max_element(model_output.begin(), model_output.end());

  float sum = 0.0f;
  for (std::size_t i = 0; i < kEmotionClassCount; ++i) {
    probs[i] = std::exp(model_output[i] - max_logit);
    sum += probs[i];
  }

  if (sum <= std::numeric_limits<float>::epsilon()) {
    probs.fill(1.0f / static_cast<float>(kEmotionClassCount));
    return probs;
  }

  for (float &value : probs) {
    value /= sum;
  }

  return probs;
}
