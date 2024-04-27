import gradio as gr
import os
import shutil
from utils import segment, plotter, writer


class GlobalState:
    """
    Class to store global variables
    """
    heart_area = [.54, .5, -.14, .16]
    smooth_factor = 3
    video_file_path = os.path.join(os.path.dirname(__file__), 'videos/')
    input_path = None
    line = None
    result_file_path = os.path.join(os.path.dirname(__file__), 'result/result.mp4')
    result_folder = os.path.join(os.path.dirname(__file__), 'result/')
    yolo_path = os.path.join(os.path.dirname(__file__), 'pretrained_yolo.pt')


def upload_video_file(fid):
    """
    uploads and save video to workdir
    """
    raw_path = os.path.join(GlobalState.video_file_path, os.path.basename(fid.name))
    shutil.move(fid.name, raw_path)
    GlobalState.input_path = raw_path
    gr.Info("Video uploaded")
    return gr.update('Run!')


def processing(sl):
    GlobalState.smooth_factor = int(sl)
    graph, frames, message = segment(GlobalState.input_path, GlobalState.yolo_path, start=0, fstep=1,
                                     crop=GlobalState.heart_area)
    gr.Info(message)
    
    if message == 'Video processing succeeded':
        GlobalState.line = graph
        writer(GlobalState.result_file_path, frames)
        gr.Info('Processed video saved!')
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)
    
    
def plot_graph(sl):
    sl = int(sl)
    result, text = plotter(GlobalState.line, sl)
    return result, gr.update(value=text)

def show_video(btn):
    return gr.update(label="Segmented Echo", value=GlobalState.result_file_path)

def main():

    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'videos/'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'result/'), ignore_errors=True)
    os.mkdir(os.path.join(os.path.dirname(__file__), 'videos/'))
    os.mkdir(os.path.join(os.path.dirname(__file__), 'result/'))

    with gr.Blocks() as demo:
        with gr.Tab("Load"):
            with gr.Row():
                gr.Markdown(
                    """
                    # Load video file 🫀
                    # Then press **Run!**
                    # Have fun:)
                    """)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        video_upload = gr.File(label="Upload heart Echo", file_types=["video"], file_count="single")
                    with gr.Row():
                        process_button = gr.Button("Run!")
                    with gr.Row():
                        player = gr.Video(label="Segmented Echo", value=None, format='mp4')

                with gr.Column():
                    with gr.Row():
                        smoother = gr.Slider(1, 50, 5, 1, label="Rolling Mean Window")
                    with gr.Row():
                        messenger = gr.Textbox(label='Ejection Fracture', value=None)
                    with gr.Row():
                        plot = gr.LinePlot(x="Frame", y="Left ventricle visible area, px*px",
                                           overlay_point=False, 
                                           tooltip=["Frame", "Left ventricle visible area, px*px"], 
                                           width=500, height=300)
                    with gr.Row():
                        show_graph = gr.Button('Plot', visible=False)
                    with gr.Row():
                        show_button = gr.Button("Show result!")

        video_upload.upload(upload_video_file, video_upload, outputs=[process_button], show_progress='full')
        process_button.click(processing, inputs=[smoother], outputs=[show_graph, show_button], show_progress='full')
        show_graph.click(plot_graph, inputs=[smoother], outputs=[plot, messenger])
        show_button.click(show_video, outputs=[player])
        player.change(show_video, outputs=[player])


    demo.launch(share=True, allowed_paths=[GlobalState.result_folder])


if __name__ == "__main__":
    main()
