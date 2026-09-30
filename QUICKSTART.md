# ⚡ QUICK START - 3 Commands to Run

Get up and running in **5 minutes** with just 3 commands!

---

## 1️⃣ Setup (2 minutes)

```bash
# Open terminal in this folder and run:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2️⃣ Download Models (2 minutes)

```bash
python scripts/download_models.py
```

---

## 3️⃣ Run Demo (1 minute)

```bash
# Get any car image and save to data/samples/test.jpg
# Then run:
python scripts/demo.py --image data/samples/test.jpg
```

---

## ✅ Done!

You should see:
```
✓ Found X vehicle(s)
✓ Result saved to: data/outputs/demo_result.jpg
```

---

## 🎯 Next Steps

### Try Different Models:

Edit `configs/model_config.yaml` and change:
```yaml
active_model: "yolov8s"  # Try: yolov8n, yolov8s, yolov8m
```

### Start API:

```bash
cd backend
uvicorn main:app --reload
# Visit: http://localhost:8000/docs
```

### Explore Notebook:

```bash
jupyter notebook notebooks/01_vehicle_detection.ipynb
```

---

## 📚 Full Documentation

- **Detailed Setup:** See `SETUP_GUIDE.md`
- **What Was Created:** See `WHAT_WAS_CREATED.md`
- **Project README:** See `README.md`

---

**That's it! You're ready to build! 🚀**
