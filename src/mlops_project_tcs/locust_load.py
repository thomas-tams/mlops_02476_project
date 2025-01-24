from locust import HttpUser, task, between
from pathlib import Path


class FastAPIUser(HttpUser):
    # Set the wait time between tasks
    wait_time = between(1, 3)

    @task
    def health_check(self):
        # Simulate a request to the health check endpoint (GET /)
        self.client.get("/")

    @task
    def classify_image(self):
        # Simulate a POST request to the /predict/ endpoint with a test image
        test_image_path = Path("tests/resources/dummy_images/yes/dummy_image_yes_1.jpg")
        with open(test_image_path, "rb") as image_file:
            self.client.post("/predict/", files={"file": (test_image_path.name, image_file, "image/jpeg")})

    @task
    def preprocess_image(self):
        test_image_path = Path("tests/resources/dummy_images/yes/dummy_image_yes_1.jpg")
        # Simulate a POST request to the /preprocess/ endpoint
        with open(test_image_path, "rb") as image_file:
            self.client.post("/preprocess/", files={"file": (test_image_path.name, image_file, "image/jpeg")})
