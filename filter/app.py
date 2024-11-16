import gradio as gr
from filter.src.UseFilter import  apply_filter_on_image, apply_filter_on_video

with gr.Blocks() as demo:
    # webcam
    with gr.Tab("Webcam Input"):
        with gr.Row():
            with gr.Column():
                filter_name = gr.Dropdown(choices=["cat_nose", "dog_nose", "cat_ears", "dog_ears", "no_mask", "Face Ronaldo", "Mask Squid Game", "Mask Anonymous", "landmarks"],
                                          value="no_mask", label="Filter")
                input_img_webcam = gr.Image(sources=["webcam"], type="numpy")
            with gr.Column():
                output_img = gr.Image(streaming=True)

            # Processing webcam image
            input_img_webcam.stream(apply_filter_on_image, [input_img_webcam, filter_name], [output_img],
                                    time_limit=30, stream_every=0.1, concurrency_limit=30)
            
    # image
    with gr.Tab("Upload Image"):
        with gr.Row():
            with gr.Column():
                filter_name_upload = gr.Dropdown(choices=["cat_nose", "dog_nose", "cat_ears", "dog_ears", "no_mask", "Face Ronaldo", "Mask Squid Game", "Mask Anonymous","landmarks"],
                                                 value="no_mask", label="Filter")
                input_img_upload = gr.Image(type="numpy", label="Upload Image")
            with gr.Column():
                output_img_upload = gr.Image()

            # Processing uploaded image
            input_img_upload.change(apply_filter_on_image, [input_img_upload, filter_name_upload], [output_img_upload])

    # video
    with gr.Tab("Upload video"):
        with gr.Row():
            with gr.Column():
                filter_name_upload = gr.Dropdown(choices=["cat_nose", "dog_nose", "cat_ears", "dog_ears", "no_mask", "Face Ronaldo", "Mask Squid Game", "Mask Anonymous", "landmarks"],
                                                 value="no_mask", label="Filter")
                input_video_upload = gr.Video(label="Upload Video")
            with gr.Column():
                output_video_upload = gr.Video()
            
            input_video_upload.change(apply_filter_on_video, [input_video_upload, filter_name_upload], [output_video_upload])
            
if __name__ == "__main__":
    demo.launch(share=True)
