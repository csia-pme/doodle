import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import (
    FieldDescriptionType,
    ExecutionUnitTagName,
    ExecutionUnitTagAcronym,
)
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
import io
import os
import torch.nn as nn
import torch
import json
import numpy as np
from PIL import Image

DOODLE_RECOGNITION_NETWORK = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "networks/simpler_long_aug.nn"
)
DOODLE_CLASSNAMES_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data/classname_EN_filtered.txt"
)

settings = get_settings()


#   //////////////  DECODER   //////////////
class SimpleDoodleClassifier(nn.Module):
    def __init__(self, nbr_classes=354):
        super(SimpleDoodleClassifier, self).__init__()

        elems = []
        elems += [
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
        ]  # 28x28
        elems += [
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
        ]  # 28x28
        elems += [
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        ]  # 14x14

        elems += [nn.Flatten()]
        elems += [nn.Linear(64 * 14 * 14, 256), nn.Dropout(), nn.LeakyReLU()]
        elems += [nn.Linear(256, 256), nn.Dropout(), nn.LeakyReLU()]
        elems += [nn.Linear(256, nbr_classes)]

        self.network = nn.Sequential(*elems)

    def forward(self, imgs):
        likelihood = self.network(imgs)

        #   stable softmax
        normalized = (
                torch.exp(likelihood - torch.max(likelihood, axis=1)[0][:, None]) + 1e-20
        )
        return normalized / torch.sum(normalized, axis=1)[:, None]


#   //////////////  DECODER   //////////////
class SimplerDoodleClassifier(nn.Module):
    def __init__(self, nbr_classes=354):
        super(SimplerDoodleClassifier, self).__init__()

        elems = []
        elems += [
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding="same"),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        ]  # 28x28
        elems += [
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding="same"),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        ]  # 28x28
        elems += [
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding="same"),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        ]  # 14x14

        elems += [nn.Flatten()]
        elems += [nn.Linear(16 * 14 * 14, 256), nn.Dropout(0.25), nn.LeakyReLU()]
        elems += [nn.Linear(256, 512), nn.Dropout(0.25), nn.LeakyReLU()]
        elems += [nn.Linear(512, nbr_classes)]

        self.network = nn.Sequential(*elems)

    def forward(self, imgs):
        likelihood = self.network(imgs)

        #   stable softmax
        normalized = (
                torch.exp(likelihood - torch.max(likelihood, axis=1)[0][:, None]) + 1e-20
        )
        return normalized / torch.sum(normalized, axis=1)[:, None]


network_models = {
    "SimpleDoodleClassifier": SimpleDoodleClassifier,
    "SimplerDoodleClassifier": SimplerDoodleClassifier,
}


# Loading the Neural Network and inferences
class TestNN:
    def __init__(self, NN_path, classnames_path):
        self.classnames_path = classnames_path

        with open(classnames_path, "r") as f:
            self.classnames = f.read().splitlines()

        data = torch.load(NN_path, map_location=torch.device("cpu"))
        self.network = network_models["SimplerDoodleClassifier"](
            nbr_classes=len(self.classnames)
        )
        self.network.load_state_dict(data["network"])
        self.network.eval()

    def infer(self, img):
        results = self.network(img)
        pairs = [
            (results[0, i].item(), self.classnames[i])
            for i in range(len(self.classnames))
        ]
        pairs.sort(key=lambda x: x[0], reverse=True)

        return pairs


class MyService(Service):
    """
    Doodle service
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger
    _network: object

    def __init__(self):
        super().__init__(
            name="Doodle",
            slug="doodle",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_PNG,
                        FieldDescriptionType.IMAGE_JPEG,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_JSON]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_RECOGNITION,
                    acronym=ExecutionUnitTagAcronym.IMAGE_RECOGNITION,
                ),
            ],
            has_ai=True
        )
        self._logger = get_logger(settings)
        self._model = TestNN(DOODLE_RECOGNITION_NETWORK, DOODLE_CLASSNAMES_PATH)

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        raw = data["image"].data
        # ... do something with the raw data
        with Image.open(io.BytesIO(raw)) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = im.resize((514, 514))
            fulls = 255 - np.asarray(im)[:, :, 0]
            rows = np.sum(fulls, axis=0)
            cols = np.sum(fulls, axis=1)

            def findFirstNonNull(elem):
                for i in range(elem.shape[0]):
                    if elem[i] != 0:
                        return i
                return None

            min_x = findFirstNonNull(rows)

            if min_x is None:
                return {
                    "result": TaskData(
                        data=json.dumps({"empty": 100.0}, ensure_ascii=False),
                        type=FieldDescriptionType.APPLICATION_JSON,
                    )
                }

            max_x = 511 - findFirstNonNull(rows[::-1])
            min_y = findFirstNonNull(cols)
            max_y = 511 - findFirstNonNull(cols[::-1])

            crop = im.crop((min_x, min_y, max_x, max_y))

            npimg = np.asarray(crop.resize((28, 28), 2))
            npimg = npimg.astype(np.float32)[:, :, 0]
            timg = torch.Tensor(npimg) / 256.0

            choice = self._model.infer(timg[None, None, :, :])

            class_labels = []
            class_likelihood = []
            cumul = 0

            for i in range(10):
                cumul += choice[i][0]
                class_likelihood.append(choice[i][0])
                class_labels.append(choice[i][1])
                if cumul > 0.9 or choice[i][0] < 0.05:
                    break

            class_likelihood.append(1 - cumul)
            class_labels.append("")
            explode = [0] * len(class_labels)
            explode[0] = 0.1

            res = {
                class_labels[i]: class_likelihood[i] for i in range(len(class_labels))
            }

            # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
            return {
                "result": TaskData(
                    data=json.dumps(res, ensure_ascii=False),
                    type=FieldDescriptionType.APPLICATION_JSON,
                )
            }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(
                    my_service, engine_url
                )
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """This service will guess what have been doodled.
"""
api_summary = """This service will guess what have been doodled.
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="Doodle API.",
    description=api_description,
    version="1.0.0",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
