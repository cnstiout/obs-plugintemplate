#include "emotion_mapping.hpp"

Emotion EmotionFromModelIndex(const std::size_t model_index)
{
  switch (model_index) {
  case 0:
    return Emotion::Neutre;
  case 1:
    return Emotion::Joie;
  case 2:
    return Emotion::Surprise;
  case 3:
    return Emotion::Tristesse;
  case 4:
    return Emotion::Colere;
  case 5:
    return Emotion::Degout;
  case 6:
    return Emotion::Peur;
  case 7:
    return Emotion::Incertain;
  default:
    return Emotion::Incertain;
  }
}

const char *EmotionToFrenchLabel(const Emotion emotion)
{
  switch (emotion) {
  case Emotion::Joie:
    return "Joie";
  case Emotion::Tristesse:
    return "Tristesse";
  case Emotion::Colere:
    return "Colere";
  case Emotion::Peur:
    return "Peur";
  case Emotion::Surprise:
    return "Surprise";
  case Emotion::Degout:
    return "Degout";
  case Emotion::Neutre:
    return "Neutre";
  case Emotion::Incertain:
  default:
    return "Incertain";
  }
}

cv::Scalar EmotionColorBgra(const Emotion emotion)
{
  switch (emotion) {
  case Emotion::Joie:
    return cv::Scalar(0, 220, 255, 255);
  case Emotion::Tristesse:
    return cv::Scalar(255, 96, 32, 255);
  case Emotion::Colere:
    return cv::Scalar(32, 32, 240, 255);
  case Emotion::Peur:
    return cv::Scalar(180, 0, 220, 255);
  case Emotion::Surprise:
    return cv::Scalar(0, 255, 120, 255);
  case Emotion::Degout:
    return cv::Scalar(80, 180, 80, 255);
  case Emotion::Neutre:
    return cv::Scalar(200, 200, 200, 255);
  case Emotion::Incertain:
  default:
    return cv::Scalar(100, 100, 100, 255);
  }
}
