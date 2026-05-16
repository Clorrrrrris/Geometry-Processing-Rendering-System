# Cloris Liu 261010926
import parser
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import taichi as ti
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument("-i", "--infile", nargs='+', type=str, help="Name of json file that will define the scene", required = True)
parse.add_argument("-o", "--outdir", type=str, default="out", help="directory for output files, default is ./out")
parse.add_argument('-s', '--show', action='store_true', help="Show the image in a window")
parse.add_argument('-f', '--factor', type=float, default=1.0, help="Scale factor for resolution")
parse.add_argument('-ti', '--taichi', type=str, default='cpu', help="Taichi backend 'cpu', 'vulkan', 'cuda', 'metal', cpu is default")

args = parse.parse_args()

def save_image( image: ti.Vector.field, scene_file_name: str, outdir_name: str ):
        img = image.to_numpy()
        # remove the path and extension from scene file, put it in outdir with png extension
        outdir = pathlib.Path(outdir_name)
        outdir.mkdir(exist_ok=True) # Create output directory if it doesn't exist
        fout = str(outdir / pathlib.Path(scene_file_name).stem) + ".png"
        print("Saving image to", fout)
        img = np.clip(img, 0, 1)           # Clamp to [0,1]
        img = np.swapaxes(img, 0, 1)       # Change shape to (H, W, 3)        
        img = np.flip(img, axis=0)         # Flip vertically (up/down)
        matplotlib.image.imsave(fout, img, vmin=0, vmax=1)


if __name__ == "__main__":

    if args.taichi == 'vulkan':
        ti.init(ti.vulkan)
    elif args.taichi == 'cuda':
        ti.init(ti.cuda)
    elif args.taichi == 'metal':
        ti.init(ti.metal)
    else:
        ti.init(ti.cpu, debug=True)

    for scene_file_name in args.infile:
        full_scene = parser.load_scene(scene_file_name, image_scale_factor=args.factor)
        print("Scene Loaded")

        if args.show:
            gui = ti.GUI('Image', (full_scene.camera.width, full_scene.camera.height)) 
            iteration = 1
            while gui.running:
                if iteration <= full_scene.samples and full_scene.samples > 0:
                    full_scene.render(iteration)
                    print(f"Completed {iteration-1} / {full_scene.samples} samples per pixel")
                    iteration += 1
                gui.set_image( full_scene.image )
                gui.show()
            save_image( full_scene.image, scene_file_name, args.outdir )
        else:
            if full_scene.samples < 0:
                full_scene.samples = 1  # just do one iteration if not showing and requesting infinite samples
            for iteration in range(1, full_scene.samples + 1):
                full_scene.render( iteration )
                print(f"Completed {iteration} / {full_scene.samples} samples per pixel")        
            save_image( full_scene.image, scene_file_name, args.outdir )
