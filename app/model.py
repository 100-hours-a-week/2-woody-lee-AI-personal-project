import torch
from torchvision import transforms, models
from torch import nn
from fastapi import HTTPException
from .config import MODEL_PATH, CLASS_NAMES


def load_model():
    try:
        # 동일한 아키텍처 재정의 (학습 시와 동일하게)
        model = models.mobilenet_v2(pretrained=False)
        # 마지막 분류 레이어 수정: 학습 시 사용한 출력 노드 수와 동일하게
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'모델 로드 실패: {str(e)}')


def get_transform():
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    return transform