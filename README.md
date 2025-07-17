# Pour lancer le projet

## Nettoyer

```bash
docker stop $(docker ps -q --filter ancestor=my-ml-app-dual) 2>/dev/null || true
docker stop aqi-app 2>/dev/null || true
docker rm aqi-app 2>/dev/null || true
```

## Build le docker

```bash
docker build -t my-ml-app-dual .
```

## Lancer le container

```bash
docker run -d -p 8080:80 --name aqi-app my-ml-app-dual
```

# Tester le model via endpoints

```bash
curl -X POST \
  -F "imageData=@image/image.jpg" \
  http://localhost:8080/image
```

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "co": 520.71,
    "no": 2.38,
    "no2": 16.28,
    "o3": 130.18,
    "so2": 47.68,
    "pm2_5": 65.96,
    "pm10": 72.13,
    "nh3": 8.36
  }' \
  http://localhost:8080/predict-aqi
  ```

# Stop le container

```bash
docker stop $(docker ps -q)
```