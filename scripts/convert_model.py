from tensorflow.keras.models import load_model

# کانورت میکنیم فرمت کراس رو چون ری اکت با تنسورفلو به کراس وصل نمیشه
model = load_model('models/modified_model.keras')

# سیو میکنیم توی فرمت کامپتیبل
model.save('models/modified_model.h5')