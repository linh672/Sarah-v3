# Sarah-v3

**Sarah-v3** is a voice-controlled AI assistant built with Python that integrates speech recognition, natural language understanding, and machine learning-based predictions. It can respond to conversational queries, tell jokes, report weather and local time, and help users predict house prices and flight delays through interactive voice-based sessions.

---

## 🔧 Features

* 🗣️ **Speech Recognition**: Converts spoken input to text using `recognize_speech()`.
* 🔊 **Text-to-Speech**: Provides verbal responses via `speak_response()`.
* 🧠 **Intent Classification**: Uses a neural network model trained on custom intents to understand general-purpose questions.
* 🏠 **House Price Prediction**: Collects inputs interactively and uses a trained regression model to estimate house prices.
* ✈️ **Flight Delay Prediction**: Interactively collects flight details and predicts whether a flight will be delayed using a classification model.
* 🌤️ **Weather Reports**: Provides current weather information for specified cities.
* 🕒 **Time & Date Lookup**: Tells the current time or date in any specified city.
* 😂 **Jokes**: Tells a random joke using the `pyjokes` library.
* 👋 **Conversational Flow**: Handles greetings, thanks, and goodbyes gracefully.

---

## 📁 Project Structure

```
Sarah-v3/
│
├── house_price_predictor/
│   └── _house_price_model.pkl         # Trained regression model with pipeline
│
├── flight_delay_predictor/
│   ├── _flight_delay_model.pkl        # Trained classification model
│   └── _label_encoder.pkl             # Label encoder for predictions
│
├── intent_base_model/
│   ├── Sarah_v3_assistant.py          # Intent classification logic
│   ├── Sarah_v3_model.pth             # Trained neural network model
│   ├── dimensions.json                # Input/output dimensions
│   └── intents.json                   # User intents and responses
│
├── sarah_module/
│   ├── speech_to_text.py              # Speech recognition functions
│   ├── text_to_speech.py              # TTS functions
│   └── basic_module.py                # Weather and time utilities
│
└── main.py                            # Core logic (your provided script)
```

---

## 📦 Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
```

**Typical dependencies include:**

* `speechrecognition`
* `pyttsx3`
* `pyjokes`
* `joblib`
* `pandas`
* `scikit-learn`
* `torch`
* `word2number`
* `requests`
* `asyncio`

---

## 🚀 Running the Assistant

Run the assistant with:

```bash
python main.py
```

Then, speak commands like:

* “Hi Sarah” or “Wake up Sarah”
* “Predict house price”
* “Predict flight delay”
* “What’s the weather in London?”
* “What time is it in New York?”
* “Tell me a joke”
* “Thank you” or “Goodbye”

---

## 🧠 Machine Learning Capabilities

* **House Price Model**: Predicts based on features like location, number of rooms, income, and proximity to ocean.
* **Flight Delay Model**: Predicts based on date/time, airline, and schedule details.
* Both use pre-trained pipelines stored as `.pkl` files for direct use without retraining.

---

## 🗂 Notes

* Ensure your microphone is set up and accessible by the system.
* The assistant handles inputs in a guided manner, asking for each required feature one by one.
* Make sure the model and data files are located in the correct paths as referenced in the code.



