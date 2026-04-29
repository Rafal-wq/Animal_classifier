# Animal Classifier

Aplikacja **desktopowa** do klasyfikacji zwierząt z wykorzystaniem sieci neuronowej MobileNetV2.  
Rozpoznawane gatunki: **pies, kot, koń, krowa, kura**.

---

## Wymagania

- Python 3.10
- pip

> ⚠️ TensorFlow 2.13 wymaga dokładnie **Python 3.10**. Wyższe wersje (3.11+) nie są obsługiwane.

---

## Instalacja

### 1. Sklonuj repozytorium

```bash
git clone https://github.com/Rafal-wq/Animal_classifier.git
cd Animal_classifier
```

### 2. Utwórz wirtualne środowisko

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Zainstaluj zależności

**macOS / Linux:**
```bash
pip3 install -r requirements.txt
```

**Windows:**
```bash
pip install -r requirements.txt
```

---

## Przygotowanie datasetu

Dataset nie jest częścią repozytorium. Pobierz go z poniższego linku i umieść w folderze `dataset/` zgodnie z poniższą strukturą:

**[Pobierz dataset (Google Drive)](https://drive.google.com/drive/folders/1HgjN2ndqeGjJsT6VassAMH4I0pOK0Tmn?usp=sharing)**

```
dataset/
├── pies/
├── kot/
├── kon/
├── krowa/
└── kura/
```

Każdy folder powinien zawierać zdjęcia danego zwierzęcia w formacie JPG, PNG lub WEBP.

---

## Trenowanie modelu

Po przygotowaniu datasetu uruchom skrypt treningowy:

**macOS / Linux:**
```bash
python3 train.py
```

**Windows:**
```bash
python train.py
```

Po zakończeniu treningu w folderze projektu pojawi się plik `animal_model.h5`.

---

## Uruchomienie aplikacji desktopowej

**macOS / Linux:**
```bash
python3 run.py
```

**Windows:**
```bash
python run.py
```

Aplikacja uruchomi się jako natywne okno desktopowe.  
Wgraj zdjęcie zwierzęcia przyciskiem **Load Image**, a następnie kliknij **Classify** — aplikacja wyświetli wynik wraz z rozkładem prawdopodobieństw dla wszystkich 5 gatunków.

> Plik `app.py` uruchamia wersję webową aplikacji dostępną pod `http://127.0.0.1:5000` — przydatny do testów i developmentu.

---

## Struktura projektu

```
Animal_classifier/
├── dataset/           # Zdjęcia treningowe (nie w repozytorium)
├── templates/
│   └── index.html     # Interfejs użytkownika (HTML/CSS/JS)
├── static/
│   └── style.css      # Style aplikacji
├── app.py             # Serwer Flask (backend)
├── run.py             # Uruchamianie aplikacji desktopowej (pywebview)
├── train.py           # Skrypt treningowy
├── class_names.txt    # Lista klas
├── animal_model.h5    # Wytrenowany model
└── requirements.txt   # Zależności
```

---

## Technologie

- [TensorFlow 2.13](https://www.tensorflow.org/) — sieć neuronowa MobileNetV2
- [Flask](https://flask.palletsprojects.com/) — backend aplikacji
- [pywebview](https://pywebview.flowrl.com/) — natywne okno desktopowe
- [Pillow](https://pillow.readthedocs.io/) — przetwarzanie obrazów
- [NumPy](https://numpy.org/) — operacje na danych

---

## Autorzy

- Sebastian Górski  
- Rafał Wilczewski  
- Jakub Grzymisławski  
- Łukasz Szenkiel  

Projekt studencki — Collegium Witelona Uczelnia Państwowa w Legnicy  
Metody Sztucznej Inteligencji, Semestr VI, 2025/2026
