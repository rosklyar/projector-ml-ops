import logging
import uuid
import boto3
import os

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import CallbackContext

from config import LABELS, BUCKET_NAME, ROOT_FOLDER

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

s3 = boto3.client('s3')

# upload config.json to s3
s3.upload_file('config.json', BUCKET_NAME, f'{ROOT_FOLDER}/config.json')

async def help(update: Update, context: CallbackContext):
    instructions_text = "Інструкція по використанню:\n" \
                        "1. Перейдіть до вибору категорії зі списку за допомогою команди \n/category.\n" \
                        "2. Після вибору категорії - зробіть фото і відправте боту. Будь-ласка, відправляйте одне фото за раз.\n" \
                        "3. Підтвердіть відправку фото, якщо переконались у його якості і правильності. Щоб надіслати інше фото - просто додайте нове. Щоб змінити категорію - натисніть \n/category.\n"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=instructions_text)


async def category(update: Update, context: CallbackContext):
    keyboard = [[f"{label}:{text}"] for (label, text) in LABELS.items()]

    reply_markup = ReplyKeyboardMarkup(keyboard)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Виберіть категорію.", reply_markup=reply_markup)

async def label_selected(update: Update, context: CallbackContext):
    splitted = update.message.text.split(":")
    if len(splitted) == 0 or splitted[0] not in LABELS.keys():
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Ми не розпізнали категорію. Спробуйте ще раз.")
        return
    context.chat_data['label'] = splitted[0]
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Ви обрали категорію: {LABELS[splitted[0]]}. Відправте фото, яке відповідає цій категорії")

async def make_photo(update: Update, context: CallbackContext):
    if context.chat_data.get('label') is None:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Спочатку оберіть категорію")
        await category(update, context)
    else:
        label = context.chat_data.get('label')
        photo = update.message.photo[-1]
        context.chat_data['photo_id'] = photo.file_id
        keyboard = [
            [InlineKeyboardButton("Підтвердити", callback_data="confirm")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Ви додали фото i обрали категорію '{LABELS[label]}'. Підтвердіть збереження або відхиліть його.", reply_markup=reply_markup)

async def callback(update: Update, context: CallbackContext):
    query = update.callback_query
    data = query.data

    if data.startswith("confirm"):
        label = context.chat_data.get('label')
        photo_id = context.chat_data.get('photo_id')
        photo = await context.bot.get_file(photo_id)
        file_name = f'{uuid.uuid4()}.jpg'
        file_path = os.path.join(os.getcwd(), file_name)
        await photo.download_to_drive(file_path)
        s3.upload_file(file_path, BUCKET_NAME, f'{ROOT_FOLDER}/{label}/{file_name}')
        await delete_file(file_path)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Фото збережено. Відправте нове фото. Якщо бажаєте відправити нову категорію - натисніть \n/category.")

async def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print("Error: %s : %s" % (file_path, e.strerror))