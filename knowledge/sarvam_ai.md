# Sarvam AI

Sarvam AI is an Indian AI company founded in 2023, headquartered in Bangalore. The company's mission is to build AI for India, with a focus on Indian languages and the specific needs of India's population.

Sarvam AI's key models include Saaras (speech-to-text), Bulbul (text-to-speech), and Sarvam-Translate for translation between Indian languages and English.

Saaras v3 is Sarvam's automatic speech recognition model. It supports all 22 scheduled Indian languages and offers several transcription modes: transcribe (standard mode), translate (translates speech directly to English), verbatim (preserves exact spoken words including repetitions), transliteration (converts speech to Roman script), and codemix (handles code-switched speech that mixes Indian languages with English).

Bulbul v3 is Sarvam's text-to-speech model. It supports 11 language codes: hi-IN (Hindi), mr-IN (Marathi), ta-IN (Tamil), te-IN (Telugu), bn-IN (Bengali), gu-IN (Gujarati), kn-IN (Kannada), ml-IN (Malayalam), pa-IN (Punjabi), od-IN (Odia), and en-IN (Indian English). The default speaker is shubh. Text input is capped at approximately 2,500 characters per request.

Sarvam's chat models are sarvam-30b (30 billion parameters, 64K context window, supports native tool calling) and sarvam-105b (flagship model, 128K context window). Both are available through an OpenAI-compatible API endpoint at api.sarvam.ai/v1. The older sarvam-m model is legacy and should not be used for new code.

Sarvam-Translate (also called Mayura) handles translation between Indian languages and English. It accepts auto as the source language code for automatic language detection. Supported language codes follow the BCP-47 format: hi-IN, ta-IN, te-IN, mr-IN, bn-IN, gu-IN, kn-IN, ml-IN, pa-IN, od-IN, en-IN.

API authentication uses the header api-subscription-key. Free API keys are available at dashboard.sarvam.ai.
