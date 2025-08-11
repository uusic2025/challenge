# UUSIC Docker Usage Guide
This guide explains how to build and run the inference for UUSIC in a Docker environment with GPU support.

## 1. Install Docker
First, install Docker Desktop (available for Windows, macOS, and Linux):

- Download: https://www.docker.com/get-started/

After installation, verify Docker is available:

```sh
docker --version
```
If it shows a version number, Docker is installed correctly.

## 2. Build the Docker Image
Assuming your project code and Dockerfile are in the same directory:

```sh
cd /path/to/docker
docker build -f Dockerfile -t [image_name] .
```

Parameters:
- `-f Dockerfile` — specify the Dockerfile to use.
- `-t [image_name]` — name the image, e.g., uusic.
- `.` — use the current directory as the build context.

Example:
```sh
docker build -f Dockerfile -t uusic .
```

## 3. Check the Docker Container
Run with GPU support:

```sh
docker run --gpus all --rm \
  -v [/path/to/input]:/input/:ro \
  -v [/path/to/output]:/output \
  -v [/path/to/json]:/input_json:ro \
  -it [image_name]
```
Parameters:
- `--gpus all` — enable all available GPUs inside the container.
- `--rm` — remove the container automatically after it stops.
- `-v /host/path:/container/path` — mount a local directory/file into the container:
  - `/input/:ro` — input image directory (read-only).
  - `/output` — output results directory.
  - `/input_json:ro` — input json file (read-only).
- `-it` — interactive mode.
- `[image_name]` — the Docker image name.

Example:
```sh
docker run --gpus all --rm -v /path/to/data/Val/:/input/:ro -v /path/to/sample_result_submission/:/output -v /path/to/data/private_val_for_participants.json:/input_json:ro -it uusic
```

## Submit the Docker Image
Once you have built your image locally, you can submit it to the testing phase in two common ways:

### Option 1: Share via Docker Hub (Recommended)
#### 1. Create a Docker Hub account
- Sign up at: https://hub.docker.com/
- Once signed in, create a new repository.

#### 2. Log in to Docker Hub
```sh
docker login
```
Enter your Docker Hub username and password when prompted.

#### 3. Tag your image
```sh
docker tag [tag_name] [your_dockerhub_username]/[image_name]:latest
```
Replace `[your_dockerhub_username]` with your actual Docker Hub username.

#### 4. Push the image to Docker Hub

```sh
docker push [your_dockerhub_username]/[image_name]:latest
```

#### 5. Share us `[your_dockerhub_username]/[image_name]:latest` through email

### Option 2: Share via .tar file (Offline or Private Transfer)
#### 1. Save the image to a .tar file

```sh
docker save -o [file_name].tar [image_name]
```
This will create a file `[file_name].tar` containing your image.


#### 2. Send the .tar file
You can send it to us via cloud storage.
