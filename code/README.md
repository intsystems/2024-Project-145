**1) Для запуска DreamBooth:**
   
Подготовить папку с картинками вида: dataset_persons/Name_Surname_gender/
example: dataset_persons/Ann_Green_woman/...jpg

Запустить dream_booth_script_dir:

bash preproc.sh
python3 script.py --save_path(путь сохранения) "generated_images/" --num_images(количество желаемых сгенерированных картинок)  5

**3) Для запуска IP-Adapter:**
   
Подготовить папку с картинкОЙ вида: dataset_persons/Name_Surname_gender/
example: dataset_persons/Ann_Green_woman/...jpg

Запустить ip_adapter_script_dir:

bash preproc.sh
python3 script.py --save_path(путь сохранения) "generated_images/" --num_images(количество желаемых сгенерированных картинок)  5

**3) Для запуска IP-AdapterMAX или IP-AdapterAVG:**
   
Подготовить папку с картинками вида: dataset_persons/Name_Surname_gender/
example: dataset_persons/Ann_Green_woman/...jpg

Запустить ip_adapter_script_max_dir или ip_adapter_script_avg_dir:

bash preproc.sh
python3 script.py --save_path(путь сохранения) "generated_images/" --num_images(количество желаемых сгенерированных картинок)  5

**4) Для обучения и запуска IP-AdapterSelf-Attention:**
   
   Обучение:
   
   4.1 Подготовить папку с картинками вида: dataset_persons_images/Name_Surname_gender/
       example: dataset_persons_images/Ann_Green_woman/...jpg
   
   4.2 Запустить llava_script_ip_adapter_dir:
   
       bash preproc.sh
       python3 script.py
   
       Данный скрипт сгенерирует папку:
       dataset_persons_json/Name_Surname_gender.json
       example: dataset_persons_json/Ann_Green_woman.json
   
   4.3 Запустить ip_adapter_attn_training_dir:
   
       bash preproc.sh(!!! флаг images_number установить равным количеству картинок в вашей папке dataset_persons_images/Name_Surname_gender/, флаги save_steps и num_train_epochs уставновите на ваше усмотрение)

       Веса модели сохранятся в папку sd_ip_adapter/checkpoint_/pytorch_model.bin

   Запуск самой модели:
   
   4.5 Запустить ip_adapter_attn_inference_dir:
   
       bash preproc.sh
       python3 script.py --weights_path(веса модели) "sd_ip_adapter/checkpoint_/pytorch_model.bin" --save_path(путь сохранения) "generated_images/" --num_images(количество желаемых сгенерированных картинок)  5

**5) Метрики FID и IS:**

   Запустить fid_script_dir:
   
   bash preproc.sh
   python3 script.py --real_path "real_images/" --generated_path "generated_images/"

   Запустить is_script_dir:
   
   bash preproc.sh
   python3 script.py --generated_path "generated_images/"
