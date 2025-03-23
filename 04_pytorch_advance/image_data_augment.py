from PIL import Image
from torchvision import transforms

# 변환 적용 방법
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)), # 이미지 크기 조절
        transforms.ToTensor(), # PIL.Image 형식을 Tensor로 변환
    ]
)

image = Image.open('./datasets/images/cat.jpg')
transformed_image = transform(image)

print(transformed_image.shape) # torch.Size([3, 512, 512])

# 회전 및 대칭

transform = transforms.Compose(
    [
        transforms.RandomRotation(degrees=30, expand=False, center=None), # 30도 회전
        transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 대칭
        transforms.RandomVerticalFlip(p=0.5), # 50% 확률로 상하 대칭
         
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)

# 자르기 및 패딩
transform = transforms.Compose(
    [
        transforms.Random(size=(512, 512)), # 512x512로 자르고 10만큼 패딩
        transforms.Pad(padding=50, fill=(127, 127, 255), padding_mode='constant')
    ]
)
    
# 변형 (기하학적 변환)
transform = transforms.Compose(
    [
        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=30),
    ]
)

# 색상 변환 및 정규화
transform = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.ToPILImage()
    ]
)

# 노이즈
import numpy as np
from PIL import Image
from torchvision import transforms
from imgaug import augmenters as iaa

class IaaTransformers:
    def __init__(self):
        self.seq = iaa.Sequential(
            [
                iaa.SaltAndPepper(p=(0.03, 0.07)),
                iaa.Rain(speed=(0.3, 0.7))
            ]
        )
        
    def __call__(self, img):
        img = np.array(img)
        augmented = self.seq.augment_image(img)
        return Image.fromarray(augmented)
    
transform = transforms.Compose([
    IaaTransformers(),
])

# 컷아웃 및 무작위 지우기
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, value=0), # 무작위 영역을 검은색으로
        transforms.RandomErasing(p=1.0, value="random"), # 무작위 영역 모자이크
        transforms.ToPILImage()
    ]
)

# 혼합(Mixup) 및 컷믹스(CutMix) 
# 혼합은 두 이미지를 섞어서 새로운 이미지를 생성 (반반 흐리게 둘 다 보이는 거)
# 컷믹스는 일부를 잘라내서 다른 이미지 위에 자연스럽게 붙이는 거 (고양이 얼굴에 개 얼굴 붙이기)

# 혼합

class Mixup :
    def __init__(self, target, scale, alpha=0.5, beta=0.5) :
        self.target = target
        self.scale = scale
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self,image) :
        image = np.array(image)
        target = self.target.resize(self.scale)
        target = np.array(target)
        mix_image = image * self.alpha + target * self.beta
        return Image.fromarray(mix_image.astype(np.uint8))

transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        Mixup(target=Image.open("./datasets/images/dog.jpg"), scale=(512, 512), alpha=0.5, beta=0.5),
    ]
)