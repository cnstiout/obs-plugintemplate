#include "tracker.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

float ComputeIoU(const cv::Rect &lhs, const cv::Rect &rhs)
{
  const auto intersection = lhs & rhs;
  if (intersection.empty()) {
    return 0.0f;
  }

  const float intersection_area = static_cast<float>(intersection.area());
  const float union_area = static_cast<float>(lhs.area() + rhs.area() - intersection.area());
  if (union_area <= 0.0f) {
    return 0.0f;
  }

  return intersection_area / union_area;
}

float Clamp01(float value)
{
  return std::max(0.0f, std::min(1.0f, value));
}

float ComputeEmaAlpha(const double dt_seconds, const float smoothing_seconds)
{
  if (smoothing_seconds <= 0.0f) {
    return 1.0f;
  }

  const double tau = std::max(0.001, static_cast<double>(smoothing_seconds));
  const double alpha = 1.0 - std::exp(-std::max(0.0, dt_seconds) / tau);
  return Clamp01(static_cast<float>(alpha));
}

std::pair<Emotion, float>
ComputeStableLabel(const std::array<float, kEmotionClassCount> &probs, const float confidence_threshold)
{
  std::size_t best_index = 0;
  float best_value = probs[0];

  for (std::size_t i = 1; i < probs.size(); ++i) {
    if (probs[i] > best_value) {
      best_value = probs[i];
      best_index = i;
    }
  }

  Emotion label = EmotionFromModelIndex(best_index);
  if (best_value < confidence_threshold) {
    label = Emotion::Incertain;
  }

  return {label, Clamp01(best_value)};
}

} // namespace

std::vector<DetectedFace> FaceTracker::Update(
  const std::vector<RawFaceDetection> &detections,
  const uint64_t timestamp_ns,
  const int max_faces,
  const float smoothing_seconds,
  const float confidence_threshold)
{
  std::vector<DetectedFace> output_faces;

  if (detections.empty()) {
    tracks_.clear();
    return output_faces;
  }

  std::vector<RawFaceDetection> limited = detections;
  std::sort(
    limited.begin(),
    limited.end(),
    [](const RawFaceDetection &lhs, const RawFaceDetection &rhs) { return lhs.bbox.area() > rhs.bbox.area(); });

  const int clamped_max_faces = std::max(1, max_faces);
  if (static_cast<int>(limited.size()) > clamped_max_faces) {
    limited.resize(static_cast<std::size_t>(clamped_max_faces));
  }

  const std::size_t track_count = tracks_.size();
  const std::size_t detection_count = limited.size();

  std::vector<int> track_to_detection(track_count, -1);
  std::vector<bool> detection_used(detection_count, false);
  std::vector<bool> track_used(track_count, false);

  constexpr float kIouThreshold = 0.2f;
  while (true) {
    float best_iou = kIouThreshold;
    int best_track = -1;
    int best_detection = -1;

    for (std::size_t track_index = 0; track_index < track_count; ++track_index) {
      if (track_used[track_index]) {
        continue;
      }

      for (std::size_t detection_index = 0; detection_index < detection_count; ++detection_index) {
        if (detection_used[detection_index]) {
          continue;
        }

        const float iou = ComputeIoU(tracks_[track_index].bbox, limited[detection_index].bbox);
        if (iou > best_iou) {
          best_iou = iou;
          best_track = static_cast<int>(track_index);
          best_detection = static_cast<int>(detection_index);
        }
      }
    }

    if (best_track < 0 || best_detection < 0) {
      break;
    }

    track_used[static_cast<std::size_t>(best_track)] = true;
    detection_used[static_cast<std::size_t>(best_detection)] = true;
    track_to_detection[static_cast<std::size_t>(best_track)] = best_detection;
  }

  std::vector<TrackState> next_tracks;
  next_tracks.reserve(limited.size());
  output_faces.reserve(limited.size());

  for (std::size_t track_index = 0; track_index < track_count; ++track_index) {
    const int detection_index = track_to_detection[track_index];
    if (detection_index < 0) {
      continue;
    }

    TrackState track = tracks_[track_index];
    const auto &detection = limited[static_cast<std::size_t>(detection_index)];

    const double dt_seconds =
      (track.last_seen_ns > 0 && timestamp_ns >= track.last_seen_ns)
        ? static_cast<double>(timestamp_ns - track.last_seen_ns) / 1'000'000'000.0
        : 1.0 / 15.0;
    const float alpha = ComputeEmaAlpha(dt_seconds, smoothing_seconds);

    for (std::size_t i = 0; i < kEmotionClassCount; ++i) {
      track.ema_probs[i] = alpha * detection.probs_raw[i] + (1.0f - alpha) * track.ema_probs[i];
    }

    track.bbox = detection.bbox;
    track.last_seen_ns = timestamp_ns;

    const auto [stable_label, stable_conf] = ComputeStableLabel(track.ema_probs, confidence_threshold);
    track.stable_label = stable_label;
    track.stable_conf = stable_conf;

    next_tracks.push_back(track);
    output_faces.push_back(
      DetectedFace {track.track_id, detection.bbox, detection.probs_raw, stable_label, stable_conf, timestamp_ns});
  }

  for (std::size_t detection_index = 0; detection_index < detection_count; ++detection_index) {
    if (detection_used[detection_index]) {
      continue;
    }

    TrackState track;
    track.track_id = next_track_id_++;
    track.bbox = limited[detection_index].bbox;
    track.last_seen_ns = timestamp_ns;
    track.ema_probs = limited[detection_index].probs_raw;

    const auto [stable_label, stable_conf] = ComputeStableLabel(track.ema_probs, confidence_threshold);
    track.stable_label = stable_label;
    track.stable_conf = stable_conf;

    next_tracks.push_back(track);
    output_faces.push_back(
      DetectedFace {track.track_id, track.bbox, limited[detection_index].probs_raw, stable_label, stable_conf,
                    timestamp_ns});
  }

  tracks_ = std::move(next_tracks);
  std::sort(
    output_faces.begin(),
    output_faces.end(),
    [](const DetectedFace &lhs, const DetectedFace &rhs) { return lhs.bbox.area() > rhs.bbox.area(); });
  return output_faces;
}

void FaceTracker::Reset()
{
  tracks_.clear();
}
