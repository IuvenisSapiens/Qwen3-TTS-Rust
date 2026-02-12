# Qwen3-TTS Rust

[简体中文](../README.md) | [English](README_EN.md) | [Français](README_FR.md)

Ce projet est une implémentation Rust haute performance de Qwen3-TTS. Les percées majeures sont la synthèse **"Pilotée par Instructions (Instruction-Driven)"** et le **"Clonage de Voix Personnalisé (Custom Speakers)"**. En s'appuyant sur la sécurité mémoire de Rust et l'inférence efficace de llama.cpp/ONNX, il offre une solution de synthèse vocale de qualité industrielle.

## 🚀 Sauts Majeurs : Instructions & Personnalisation

Contrairement aux systèmes TTS traditionnels, Qwen3-TTS Rust vous permet de contrôler le style de parole via de simples instructions textuelles et de cloner n'importe quelle voix en quelques secondes.

### 1. Synthèse Pilotée par Instructions (Instruction-Driven)
Vous pouvez inclure des instructions d'émotion, de vitesse ou de style directement dans le texte. Le modèle de langage (LLM) utilise sa compréhension sémantique pour "savoir" comment lire.
> **Exemple**: `cargo run --example qwen3-tts -- --text "[Joyeusement] Bonjour ! Le temps aujourd'hui est absolument fantastique !" --voice-file "speaker.json"`

### 2. Voix Personnalisées (Custom Speakers)
Ne soyez plus limité aux voix prédéfinies. Avec un seul **audio de référence WAV en 24kHz**, vous pouvez créer un pack vocal unique.
-   **Extraction en un clic**: Extrait automatiquement les vecteurs du locuteur (Speaker Embedding) et les caractéristiques acoustiques (Codec Codes).
-   **Sauvegarde Permanente**: Sauvegardé en `.json` après extraction, aucun audio original n'est nécessaire pour une utilisation future.

## 🌟 Avantages Techniques

-   **Multi-Plateforme/Backends**: Adaptation profonde pour **Windows / Linux / macOS**, supportant **CPU / CUDA / Vulkan / Metal**.
-   **Runtime Sans Configuration**: Gère automatiquement les dépendances binaires de `llama.cpp` (b7885) et `onnxruntime`, avec mappage d'actifs multi-plateforme et chargement dynamique.
-   **Moteur Hybride**: 
    -   **Inférence LLM**: Utilise llama.cpp pour la conversion texte en caractéristiques acoustiques, avec accélération matérielle **Vulkan** activée par défaut.
    -   **Décodage Audio**: Utilise ONNX Runtime (CPU) pour un décodage fluide, assurant une latence minimale.

## 🛠️ Guide Rapide

### Créer et Sauvegarder une Voix Personnalisée
```powershell
cargo run --example qwen3-tts -- `
    --model-dir models `
    --ref-audio "path/to/me.wav" `
    --ref-text "Le texte prononcé pendant l'enregistrement" `
    --save-voice "models/presets/my_voice.json" `
    --text "[Excité] Hé ! Ma voix a été clonée dans le moteur Rust !" `
    --max-steps 512
```

## 📂 Gestion Automatisée
Le programme intègre une logique d'**auto-téléchargement des modèles et des runtimes**. Lors du premier lancement, il téléchargera automatiquement les modèles depuis HuggingFace et les binaires officiels de `llama.cpp` appropriés dans le dossier `runtime/` selon votre système d'exploitation.

## 📜 Licence & Remerciements
- Licence **MIT / Apache 2.0**.
- Merci au dépôt officiel [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) pour les modèles et la base technique.
- Merci à [Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF) pour l'inspiration sur le flux d'inférence GGUF.
