#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "emotion_mapping.hpp"

struct DetectedFace {
  int track_id = -1;
  cv::Rect bbox;
  std::array<float, kEmotionClassCount> probs_raw {};
  Emotion label = Emotion::Incertain;
  float confidence = 0.0f;
  uint64_t ts_ns = 0;
};

struct TrackState {
  int track_id = -1;
  cv::Rect bbox;
  std::array<float, kEmotionClassCount> ema_probs {};
  Emotion stable_label = Emotion::Incertain;
  float stable_conf = 0.0f;
  uint64_t last_seen_ns = 0;
};

struct RawFaceDetection {
  cv::Rect bbox;
  std::array<float, kEmotionClassCount> probs_raw {};
};

class FaceTracker {
public:
  std::vector<DetectedFace> Update(
    const std::vector<RawFaceDetection> &detections,
    uint64_t timestamp_ns,
    int max_faces,
    float smoothing_seconds,
    float confidence_threshold);

  void Reset();

private:
  int next_track_id_ = 1;
  std::vector<TrackState> tracks_;
};
