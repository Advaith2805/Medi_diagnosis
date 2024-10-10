import streamlit as st
from PIL import Image
from pytorch_lightning import LightningModule
from torchvision import transforms

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np 
import io
import cv2





st.set_page_config(layout="wide")


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background: rgb(34,193,195);
background: linear-gradient(0deg, rgba(34,193,195,1) 0%, rgba(45,253,90,0.8436624649859944) 100%);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
class covid_ctModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(512, 1)


        # Initialize feature_map in init but don't assign the layers yet
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, data):

        feature_map = self.feature_map(data)
        avg_pool_output = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
        avg_output_flattened = torch.flatten(avg_pool_output)
        pred = self.model.fc(avg_output_flattened)

        return pred, feature_map 


model_path="epoch_29.ckpt"
model = covid_ctModel.load_from_checkpoint(model_path,strict=False)
model.eval()
class CovidModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features = 512, out_features = 1)
        
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])
        
    def forward(self, data):
        feature_map = self.feature_map(data)
        avg_pool_output = torch.nn.functional.adaptive_avg_pool2d(input=feature_map, output_size=(1,1))
        avg_output_flattened = torch.flatten(avg_pool_output)
        pred = self.model.fc(avg_output_flattened)
        return pred, feature_map


model1_path="epoch_33.ckpt"
model1 = CovidModel.load_from_checkpoint(model1_path,strict=False)
model1.eval()

class PneumoniaModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features = 512, out_features = 1)
        
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])
        
    def forward(self, data):
        feature_map = self.feature_map(data)
        avg_pool_output = torch.nn.functional.adaptive_avg_pool2d(input=feature_map, output_size=(1,1))
        avg_output_flattened = torch.flatten(avg_pool_output)
        pred = self.model.fc(avg_output_flattened)
        return pred, feature_map

model2_path="epoch30.ckpt"
model2=PneumoniaModel.load_from_checkpoint(model2_path,strict=False)
model2.eval()



def cam(model, img):
    with torch.no_grad():       
        pred, features = model(img.unsqueeze(0))
    features = features.reshape((512, 49))
    weights_params = list(model.model.fc.parameters())[0]
    weight = weights_params[0].detach()
    
    cam = torch.matmul(weight, features)
    cam_img = cam.reshape(7,7).cpu()
    return cam_img, torch.sigmoid(pred)

def cam_ct(model,img):
  with torch.no_grad():
    pred,features=model(img.unsqueeze(0))
  features=features.reshape((512,49))
  weight_params=list(model.model.fc.parameters())[0]
  weight=weight_params[0].detach()


  cam=torch.matmul(weight,features)
  cam_img=cam.reshape(7,7).cpu()
  return cam_img,torch.sigmoid(pred)

def cam_pne(model,img):
  with torch.no_grad():
    pred,features=model(img.unsqueeze(0))
  features=features.reshape((512,49))
  weight_params=list(model.model.fc.parameters())[0]
  weight=weight_params[0].detach()


  cam=torch.matmul(weight,features)
  cam_img=cam.reshape(7,7).cpu()
  return cam_img,torch.sigmoid(pred)



     


def visualize_ct(img,cam,pred):
  img = img[0]
  cam= transforms.functional.resize(cam.unsqueeze(0), (224, 224))[0]
    
  fig, axes = plt.subplots(1, 2)
  axes[0].imshow(img, cmap="bone")
  
  axes[1].imshow(img, cmap="bone")
  axes[1].imshow(cam, alpha=0.5, cmap="jet")
  plt.title('Positive' if pred >0.5 else 'Negative')
  axes[0].axis("off")
  axes[1].axis("off")
  if pred > 0.5:
    st.write("## :red[COVID POSITIVE]")
    st.write("## You have been tested positive for COVID 19.Please follow these steps immediately:")
    st.write("- Know the full range of symptoms of COVID-19. The most common symptoms of COVID-19 are fever, dry cough, tiredness and loss of taste or smell. Less common symptoms include aches and pains, headache, sore throat, red or irritated eyes, diarrhoea,  a skin rash or discolouration of fingers or toes.")
    st.write("- Stay home and self-isolate for 10 days from symptom onset, plus three days after symptoms cease. Call your health care provider or hotline for advice. Have someone bring you supplies. If you need to leave your house or have someone near you, wear a properly fitted mask to avoid infecting others.")
    st.write("- Keep up to date on the latest information from trusted sources, such as WHO(World Health Organization) or your local and national health authorities. Local and national authorities and public health units are best placed to advise on what people in your area should be doing to protect themselves.")
  

 

  plt.show()
  st.set_option('deprecation.showPyplotGlobalUse', False)

def visualise_xray(img, cam, pred):
    img=img[0]
    
    cam = transforms.functional.resize(cam.unsqueeze(0), (224, 224))[0]
    cam = np.squeeze(cam)  # Remove extra dimensions if present
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='bone')
    axes[1].imshow(img, cmap='bone')    
    axes[1].imshow(cam, alpha=0.5, cmap='jet')    
    plt.title(f"Covid: {str(pred < 0.5)}")
    
    
    axes[0].axis("off")
    axes[1].axis("off")

    if pred <0.5:
     st.write("## :red[COVID POSITIVE]")
     st.write("## :red[Please proceed to our CT scan services immediately]") 

   

 

    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)

def visualize_pne(img,cam,pred):
  img = img[0]
  cam= transforms.functional.resize(cam.unsqueeze(0), (224, 224))[0]
  cam = np.squeeze(cam)

  fig, axes = plt.subplots(1, 2)
  axes[0].imshow(img, cmap="bone")
  
  axes[1].imshow(img, cmap="bone")
  axes[1].imshow(cam, alpha=0.5, cmap="jet")
  plt.title('Positive' if pred>0.4 else 'Negative')
  axes[0].axis("off")
  axes[1].axis("off")
  if pred >0.4:
    st.write("## :red[PNEUMONIA DETECTED]")
    st.write("## :red[Please proceed to our COVID X-Ray or CT services immediately]") 
    
   

 

  plt.show()
  st.set_option('deprecation.showPyplotGlobalUse', False)



def load_file(path):
    return np.load(path).astype(np.float32)

def xray_transform(image):
    transform=torchvision.transforms.Compose([
                                
                                transforms.ToTensor(),
                                transforms.Normalize(0.223, 0.130),

])
    return transform(image)

def ct_transform(image):
    transform=torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.144,0.149),
    ])
    return transform(image)


def pne_transform(image):
    transform=torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.49,0.248)
    ])
    return transform(image)









def pneumonia_upload_page():
    st.title("Upload Pneumonia Diagnosis File")
    uploaded_file = st.file_uploader("Choose a Pneumonia X-Ray image", type=["jpg", "png", "npy"])

    if uploaded_file is not None:
        if uploaded_file.type.startswith("image"):

            
            image = Image.open(io.BytesIO(uploaded_file.read()))

            st.image(image, caption='Uploaded Image')
            img = np.array(image)
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = img_gray / img_gray.max()
            img_gray = cv2.resize(img_gray, (224, 224)).astype(np.float32)
            img_preprocessed = pne_transform(img_gray)
            cam_img, pred = cam_pne(model2, img_preprocessed)
            cam_visualized = visualize_pne(img_preprocessed, cam_img, pred.item())
            st.write(f"Prediction: {'Positive' if pred.item() > 0.4 else 'Negative'}")
            st.pyplot(cam_visualized)

        elif uploaded_file.type == "application/octet-stream":
            
            img = load_file(uploaded_file)
            img_preprocessed = pne_transform(img)
            
            cam_img, pred = cam_pne(model2, img_preprocessed)
            cam_visualized = visualize_pne(img_preprocessed, cam_img, pred.item())
            st.write(f"Prediction: {'Positive' if pred.item() > 0.4 else 'Negative'}")
            st.pyplot(cam_visualized)


    
def xray_upload_page():

    
    st.title("Upload COVID X-Ray Diagnosis File")
    uploaded_file = st.file_uploader("Choose a COVID X-Ray image", type=["jpg", "png", "npy"])
    if uploaded_file is not None:
        if uploaded_file.type.startswith("image"):

            # Handle image files
            image = Image.open(io.BytesIO(uploaded_file.read()))

            st.image(image, caption='Uploaded Image')
            img = np.array(image)
            # Convert image to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = img_gray / img_gray.max()
            img_gray = cv2.resize(img_gray, (224, 224)).astype(np.float32)
            img_preprocessed = pne_transform(img_gray)
            cam_img, pred = cam(model1, img_preprocessed)
            cam_visualized = visualise_xray(img_preprocessed, cam_img, pred.item())
            
            st.write(f"Prediction: {'Positive' if pred.item() <0.4 else 'Negative'}")
            st.pyplot(cam_visualized)
        
        elif uploaded_file.type == "application/octet-stream":
        
         img = load_file(uploaded_file)
         img_preprocessed = xray_transform(img)
            
         cam_img, pred = cam(model1, img_preprocessed)
         cam_visualized = visualise_xray(img_preprocessed, cam_img, pred.item())
         st.write(f"Prediction: {'Positive' if pred.item() <0.4 else 'Negative'}")
         st.pyplot(cam_visualized)
        

    

def ct_upload_page():
    st.title("Upload COVID CT Diagnosis File")
    uploaded_file = st.file_uploader("Choose a COVID CT image", type=["jpg", "png", "npy"])

    if uploaded_file is not None:
         if uploaded_file.type.startswith("image"):
            image = Image.open(io.BytesIO(uploaded_file.read()))

            st.image(image, caption='Uploaded Image')
    
    
            if len(image.mode) == 1:  
                img = np.array(image)
            else:
                img = np.array(image.convert('L'))  # Convert to grayscale if not already
    
  
            img_normalized = img / img.max()
            img_resized = cv2.resize(img_normalized, (224, 224)).astype(np.float32)
    
    
            img_preprocessed = ct_transform(img_resized)
    
    
            cam_img, pred = cam_ct(model, img_preprocessed)
    
    
            cam_visualized = visualize_ct(img_preprocessed, cam_img, pred.item())
    
    
            st.write(f"Prediction: {'Positive' if pred.item() > 0.5 else 'Negative'}")
    
 
            st.pyplot(cam_visualized)

  
    
         elif uploaded_file.type == "application/octet-stream":
            
            img = load_file(uploaded_file)
            img_preprocessed = ct_transform(img)
            
            cam_img, pred = cam_ct(model, img_preprocessed)
            cam_visualized = visualize_ct(img_preprocessed, cam_img, pred.item())
            st.write(f"Prediction: {'Positive' if pred.item() > 0.5 else 'Negative'}")
            st.pyplot(cam_visualized)
    

   





def main():
    
    page = st.sidebar.selectbox("Select a Diagnosis Type", ["Pneumonia Diagnosis", "COVID X-Ray Diagnosis", "COVID CT Diagnosis"])
    
    
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = page
    
    if st.sidebar.button("Go to Selected Page"):
        st.session_state.selected_page = page

    if st.session_state.selected_page == "Pneumonia Diagnosis":
        pneumonia_upload_page()
    elif st.session_state.selected_page == "COVID X-Ray Diagnosis":
        xray_upload_page()
    elif st.session_state.selected_page == "COVID CT Diagnosis":
        ct_upload_page()

if __name__ == "__main__":
    main()