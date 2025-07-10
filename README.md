# Pour lancer le projet 

## Build le docker

```bash
docker build -t my-ml-app .
```

## Lancer le container

```bash
docker run -p 8080:80 my-ml-app
```

# Tester le model via endpoints

```bash
curl -X POST \
  -F "imageData=@image/image.jpg" \
  http://localhost:8080/image
```

# Stop le container

```bash
docker stop $(docker ps -q)
```