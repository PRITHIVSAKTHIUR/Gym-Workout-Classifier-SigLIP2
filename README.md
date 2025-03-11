![sdfvdfv.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/kJ1YXUFjmOahI6LmpI--x.png)

# **Gym-Workout-Classifier-SigLIP2**

> **Gym-Workout-Classifier-SigLIP2** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify gym workout exercises using the **SiglipForImageClassification** architecture.  

```py
    Classification Report:
                         precision    recall  f1-score   support
    
    barbell biceps curl     0.9613    0.9574    0.9593       493
            bench press     0.9402    0.9359    0.9381       437
      chest fly machine     0.9694    0.9484    0.9588       368
               deadlift     0.9833    0.9542    0.9685       371
    decline bench press     0.9884    0.9499    0.9688       359
            hammer curl     0.9917    0.9398    0.9651       382
             hip thrust     0.9692    0.9717    0.9705       389
    incline bench press     0.9297    0.9588    0.9440       510
           lat pulldown     0.9607    0.9735    0.9670       452
         lateral raises     0.9539    0.9814    0.9674       590
          leg extension     0.9573    0.9854    0.9712       410
             leg raises     0.9939    0.9109    0.9506       359
                  plank     0.9828    0.9856    0.9842       695
                pull up     0.9882    0.9744    0.9813       430
                push up     0.9382    0.9762    0.9568       420
      romanian deadlift     0.9617    0.9716    0.9667       388
          russian twist     0.8702    0.9918    0.9270       365
         shoulder press     0.9499    0.9525    0.9512       358
                  squat     0.9761    0.9441    0.9598       519
              t bar row     0.9806    0.9743    0.9774       467
            tricep dips     0.9834    0.9713    0.9773       488
        tricep pushdown     0.9837    0.9657    0.9746       437
    
               accuracy                         0.9638      9687
              macro avg     0.9643    0.9625    0.9630      9687
           weighted avg     0.9647    0.9638    0.9639      9687
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/czlXLMN1O6q2yL4364ySm.png)

The model categorizes images into 22 workout classes:
- **Class 0:** "barbell biceps curl"
- **Class 1:** "bench press"
- **Class 2:** "chest fly machine"
- **Class 3:** "deadlift"
- **Class 4:** "decline bench press"
- **Class 5:** "hammer curl"
- **Class 6:** "hip thrust"
- **Class 7:** "incline bench press"
- **Class 8:** "lat pulldown"
- **Class 9:** "lateral raises"
- **Class 10:** "leg extension"
- **Class 11:** "leg raises"
- **Class 12:** "plank"
- **Class 13:** "pull up"
- **Class 14:** "push up"
- **Class 15:** "romanian deadlift"
- **Class 16:** "russian twist"
- **Class 17:** "shoulder press"
- **Class 18:** "squat"
- **Class 19:** "t bar row"
- **Class 20:** "tricep dips"
- **Class 21:** "tricep pushdown"

# **Dataset ID2LABEL**

```py
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("YOUR-DATASET-HERE")

# Extract unique labels
labels = dataset["train"].features["label"].names

# Create id2label mapping
id2label = {str(i): label for i, label in enumerate(labels)}

# Print the mapping
print(id2label)
```

# **Run with Transformers🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Gym-Workout-Classifier-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def workout_classification(image):
    """Predicts workout exercise classification for an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "barbell biceps curl", "1": "bench press", "2": "chest fly machine", "3": "deadlift",
        "4": "decline bench press", "5": "hammer curl", "6": "hip thrust", "7": "incline bench press",
        "8": "lat pulldown", "9": "lateral raises", "10": "leg extension", "11": "leg raises",
        "12": "plank", "13": "pull up", "14": "push up", "15": "romanian deadlift",
        "16": "russian twist", "17": "shoulder press", "18": "squat", "19": "t bar row",
        "20": "tricep dips", "21": "tricep pushdown"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=workout_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Gym Workout Classification",
    description="Upload an image to classify the workout exercise."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

# **Intended Use:**  

The **Gym-Workout-Classifier-SigLIP2** model is designed to classify different gym exercises based on images. Potential use cases include:  

- **Workout Tracking:** Identifying exercises performed during a workout session.
- **Personal Training Assistance:** Helping trainers analyze and correct exercise form.
- **Gym Activity Monitoring:** Automating exercise logging and analysis in fitness apps.
- **AI-Powered Fitness Coaching:** Supporting AI-based fitness programs with real-time workout recognition.
