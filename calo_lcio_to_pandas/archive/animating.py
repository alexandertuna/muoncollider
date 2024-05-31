import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():

    images = []
    fig_anim, ax_anim = plt.subplots()

    for layer in range(10):
        ncols = 5
        cbar = [None]*ncols
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(24, 4))
        for i_ax in range(ncols):
            if i_ax == ncols - 1:
                arr = np.random.randn(10, 10)
                im = ax[i_ax].imshow(
                    np.random.randn(10, 10),
                    animated=True,
                )
                im_anim = ax_anim.imshow(
                    np.random.randn(10, 10),
                    animated=True,
                )
                cbar[i_ax] = fig.colorbar(im, ax=ax[i_ax])
                images.append([im_anim])

    if images:
        print(f"Making animation ...")
        # fig, ax = plt.subplots()
        ani = animation.ArtistAnimation(
            fig_anim,
            images,
            interval=50,
            blit=True,
            repeat_delay=1000,
        )
        print(f"Saving animation ...")
        ani.save("movie.gif")
        print(f"Done animating!")

if __name__ == "__main__":
    main()
