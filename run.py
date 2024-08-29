# %% Initialize the drawing window
import cv2
import numpy as np

image = np.array([0]*28*28*3*121, dtype=np.uint8).reshape(28*11,28*11,3)

# %% Mouse callback function
drawing = False # true if mouse is pressed
 
def draw_circle(event,x,y,flags,param):
    global drawing, image
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
 
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(image,(x-5,y-5),(x+5,y+5),(255,255,255),-1)
 
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image,(x-5,y-5),(x+5,y+5),(255,255,255),-1)
    
    elif event == cv2.EVENT_RBUTTONDOWN: #Reset image
        image = np.array([0]*28*28*3*121, dtype=np.uint8).reshape(28*11,28*11,3)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

# %% Prepare trained model
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1) # 28x28x1 -> 26x26x16
        #pool -> 13x13x16
        self.conv2 = nn.Conv2d(16, 32, 3, 1) # 13x13x16 -> 11x11x32
        #pool -> 5x5x32
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5*5*32, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)
        self.relu = nn.ReLU()
    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.flatten(X)
        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        X = self.fc3(X)
        return X
    
model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# %% Transform image to feed to model
def transform(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = torch.tensor(image, dtype=torch.float32)
    return image

# %% Feed image to model
def feed(image):
    image = image.unsqueeze(0).unsqueeze(0).float()
    return model(image)

# %% Run loop
i = 0
while True:
    cv2.imshow('image', image)
    guess_img = image.copy()
    guess_img = transform(guess_img)
    with torch.no_grad():
        guess = np.argmax(feed(guess_img))
        if i % 100 == 0: print(int(guess))
        #print guess on the image with delay
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1

cv2.destroyAllWindows()
# %%
