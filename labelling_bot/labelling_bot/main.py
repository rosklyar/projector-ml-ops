import os

from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters

from bot import category, help, callback, make_photo, label_selected

def main():
    api_token = os.getenv("API_TOKEN")
    
    application = ApplicationBuilder().token(api_token).build()
    
    application.add_handler(CommandHandler('help', help))
    application.add_handler(CommandHandler('category', category))
    application.add_handler(MessageHandler(filters=filters.TEXT & ~filters.COMMAND, callback=label_selected))
    application.add_handler(CallbackQueryHandler(callback))
    application.add_handler(MessageHandler(filters=filters.PHOTO, callback=make_photo))

    application.run_polling()

if __name__ == '__main__':
    main()