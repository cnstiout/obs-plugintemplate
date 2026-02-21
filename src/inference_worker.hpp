#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect/face.hpp>

#include "tracker.hpp"

class InferenceWorker {
public:
  struct Config {
    int max_faces = 3;
    int inference_width = 640;
    float confidence_threshold = 0.30f;
    float smoothing_seconds = 0.6f;
  };

  InferenceWorker();
  ~InferenceWorker();

  bool Start(const std::string &face_model_path, const std::string &emotion_model_path, const Config &config, std::string *error);
  void Stop();

  void UpdateConfig(const Config &config);
  void SubmitFrame(const cv::Mat &bgr_frame, uint64_t timestamp_ns, int source_width, int source_height);

  bool TryConsumeLatest(std::vector<DetectedFace> *faces, double *inference_ms, uint64_t *timestamp_ns);
  std::size_t QueueSize() const;
  bool IsRunning() const;

private:
  struct FrameTask {
    cv::Mat bgr_frame;
    uint64_t timestamp_ns = 0;
    int source_width = 0;
    int source_height = 0;
  };

  struct ResultPacket {
    std::vector<DetectedFace> faces;
    double inference_ms = 0.0;
    uint64_t timestamp_ns = 0;
    bool valid = false;
  };

  void WorkerLoop();
  std::vector<DetectedFace> RunInference(const FrameTask &task);
  std::array<float, kEmotionClassCount> InferEmotion(const cv::Mat &face_bgr);
  static std::array<float, kEmotionClassCount> NormalizeEmotionOutput(
    const std::array<float, kEmotionClassCount> &model_output);
  static bool LooksLikeProbabilities(const std::array<float, kEmotionClassCount> &values);

  mutable std::mutex config_mutex_;
  Config config_;

  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<FrameTask> queue_;

  mutable std::mutex result_mutex_;
  ResultPacket latest_result_;

  std::thread worker_thread_;
  bool stop_requested_ = false;
  std::atomic<bool> running_ {false};

  cv::Ptr<cv::FaceDetectorYN> face_detector_;
  cv::dnn::Net emotion_net_;
  FaceTracker tracker_;
};
