# InteractDiffusion for Automatic1111 Stable Diffusion WebUI
Warning: experimental inital implementation, feel free to provide feedback and pull request to improve this extension.

This extension is designed for AUTOMATIC1111's Stable Diffusion web UI, enabling the integration of InteractDiffusion with the original Stable Diffusion model. This integration enhances the capability of the web UI to control the interactions in image generation. It works with DreamBooth and LoRA models.

https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/7ea6d9f5-197a-46b2-b3c5-df659aa97f07


# Installation
1. Install the extension via link.

    Open "Extensions" tab and then select "Install from URL". Enter https://github.com/jiuntian/sd-webui-interactdiffusion.git to "URL for extension's git repository" and click Install. Finally, click "Apply and restart UI".
2. Download the model at [HuggingFace Hub](https://huggingface.co/jiuntian/interactiondiffusion-weight/blob/main/ext_interactdiff_v1.2.pth) with name `ext_interactdiff_v1.2.pth`.
3. Put models in "stable-diffusion-webui\extensions\sd-webui-interactdiffusion\models". 

# How to Use
1. Enable the extension by checking the `Enabled`.
2. Please append comma seperator to end of each entry in grounding instructions like "tohru=feeding=cat;a=doing=b".
3. Click on `Create Drawing Canvas` to create a new canvas.
4. Draw correspinding bounding boxes for subject and object, and verify the interactions on the right side.
5. Adjust scheduling sampling when necessarily.
6. Generate the image as usual.

# Gallery
&nbsp;| &nbsp;| &nbsp;| &nbsp;
--- | --- | --- | ---
![image (7)](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/e4ff1279-1b08-41c9-9ea3-45ec3667115e) | ![image (5)](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/dfd254ea-f6fb-4fc4-9fe6-8222fe47ee12) | ![image (6)](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/a6df1288-3315-4738-9db8-d9cb9bd01038) | ![image (4)](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/1766e775-ce6c-4705-a376-4aa8e62bcceb)
![cuteyukimix_1](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/1416f2b6-4907-4ac7-bb03-b5d2b5adcd91)|![cuteyukimix_7](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/7b619e4e-7d0b-4989-85f9-422fbd6a6319)|![darksushimix_1](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/2b81abe3-a39a-4db8-9e7a-63336f96d7e3)|![toonyou_6](https://github.com/jiuntian/sd-webui-interactdiffusion/assets/13869695/ce027fac-7840-44cc-9f69-0bdeef5da1da)



# Limitations
1. We currently do not support SDXL yet due to limited computation resources.
2. It could possibly cause conflicts with other extensions.
3. Some artefacts could happens on some LoRA models.

# Related Projects
This implementation is constructed based on the foundation of [sd_webui_gligen](https://github.com/ashen-sensored/sd_webui_gligen).
