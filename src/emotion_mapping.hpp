#pragma once

#include <cstddef>

#include <opencv2/core.hpp>

enum class Emotion {
  Joie = 0,
  Tristesse = 1,
  Colere = 2,
  Peur = 3,
  Surprise = 4,
  Degout = 5,
  Neutre = 6,
  Incertain = 7,
};

constexpr std::size_t kEmotionClassCount = 8;

Emotion EmotionFromModelIndex(std::size_t model_index);
const char *EmotionToFrenchLabel(Emotion emotion);
cv::Scalar EmotionColorBgra(Emotion emotion);
