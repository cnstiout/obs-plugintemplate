#include <obs-module.h>
#include <plugin-support.h>

#include "face_emotion_filter.hpp"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "fr-FR")

MODULE_EXPORT const char *obs_module_description(void)
{
  return "Filtre OBS de tracking visage et emotions (offline).";
}

bool obs_module_load(void)
{
  obs_register_source(GetFaceEmotionFilterSourceInfo());
  obs_log(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
  return true;
}

void obs_module_unload(void)
{
  obs_log(LOG_INFO, "plugin unloaded");
}
