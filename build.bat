echo Nettoyer les images et conteneurs inutilisés
docker build -t mlos-classifier .
docker run --shm-size=4g -v C:\Users\girau\Documents\Oulu\dl_final_proj\src\data:/app/src/data -v C:\Users\girau\Documents\Oulu\dl_final_proj\src\models:/app/src/models -v C:\Users\girau\Documents\Oulu\dl_final_proj\src\results:/app/src/results mlos-classifier python run.py --eval
