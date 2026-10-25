// japanese_phonemizer_stub.cpp - link-time stand-in for upstream's
// OpenJTalk-backed JapanesePhonemizer (libpiper/src/japanese_phonemizer.cpp
// in piper1-gpl 74ac24e, added by PR #325 between the 404aefe and 74ac24e
// pins).
//
// The real implementation needs the OpenJTalk C library (MeCab/NJD/JPCommon),
// which upstream builds via ExternalProject and which is not yet ported to
// this Android build. Until it is, Japanese voices must fail creation
// cleanly: piper.cpp constructs the phonemizer only for phoneme_type
// "japanese" voices and treats load()==false as a creation failure
// (nullptr -> PiperException in Kotlin), which is exactly the behaviour
// upstream specifies for a missing dictionary. Non-Japanese voices never
// touch this class beyond the unique_ptr member in piper_synthesizer.
//
// Do NOT ship Japanese voice support on top of this stub: load() always
// returns false by design.

#include "japanese_phonemizer.h"

namespace piper {

JapanesePhonemizer::JapanesePhonemizer() = default;
JapanesePhonemizer::~JapanesePhonemizer() = default;

bool JapanesePhonemizer::load(const std::string & /*openjtalk_dict_dir*/) {
  return false;
}

std::vector<std::string>
JapanesePhonemizer::phonemize(const std::string & /*text*/) {
  return {};
}

std::vector<std::string>
JapanesePhonemizer::splitSentences(const std::string &text) {
  if (text.empty()) return {};
  return {text};
}

std::string
JapanesePhonemizer::phonemizeSentence(const std::string & /*sentence*/) {
  return {};
}

}  // namespace piper
