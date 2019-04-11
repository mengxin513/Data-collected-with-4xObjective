
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

def plotting1(x_data, y_data, x_label, y_label):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_data, y_data, "r+")
    ax.set_aspect(1)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    r = max(xmax - xmin, ymax - ymin)
    ax.set_xlim((xmax + xmin)/2 - r/2, (xmax + xmin)/2 + r/2)
    ax.set_ylim((ymax + ymin)/2 - r/2, (ymax + ymin)/2 + r/2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

def plotting2(x_data, y_data, x_label, y_label, group):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_data, y_data, "+-")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.suptitle(group.name[1:])

    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

if __name__ == "__main__":

    with PdfPages("raster_snake.pdf") as pdf:

        print ("Loading data...")

        microns_per_pixel = 4.25
        df = h5py.File("raster.hdf5", mode = "r")
        for i in range(len(df)):
            group = df["raster%03d" % i]
            subgroup = group["snake_raster000"]
            n = len(subgroup)
            data = np.zeros([n, 6])
            for i in range(n):
                dset = subgroup["data%03d" % i]
                data[i, 0:3] = dset["cam_position"]
                data[i, 3:6] = dset["stage_position"]

            t = data[:, 0]
            pixel_shifts = np.zeros([n, 3])
            location_shifts = np.zeros([n, 3])

            for i in range(n):
                pixel_shifts[i, 0] = data[i, 1] - np.mean(data[:, 1])
                pixel_shifts[i, 1] = data[i, 2] - np.mean(data[:, 2])
                pixel_shifts[i, 2] = 1
                location_shifts[i, 0] = data[i, 3] - np.mean(data[:, 3])
                location_shifts[i, 1] = data[i, 5] - np.mean(data[:, 5])
                location_shifts[i, 2] = 1
                
            # exclude points at the extreme of Y (the image analysis broke)
            # the first 3 points look dodgy, and we think they are at -14500
            # up to -12500, so let's exclude all points with stage y less
            # than -12000
            #stage_dy = location_shifts[:, 1]
            #stage_dy -= np.mean(stage_dy)
            #mask = stage_dy > -12000
            #masked_location_shifts = np.empty((int(np.sum(mask)), 3), dtype=np.float)
            #masked_pixel_shifts = np.empty_like(masked_location_shifts)
            #for i, j in enumerate(np.nonzero(mask)[0]):
            #    masked_location_shifts[i,:] = location_shifts[j,:]
            #    masked_pixel_shifts[i,:] = pixel_shifts[j,:]
            #location_shifts = masked_location_shifts
            #pixel_shifts = masked_pixel_shifts

            A, res, rank, s = np.linalg.lstsq(location_shifts, pixel_shifts)
            #A is the least squares solution pixcel_shifts*A = location_shifts
            #res is the sums of residuals location_shifts - pixcel_shifts*A
            #rank is rank of matrix pixcel_shifts
            #s is singular values of pixcel_shifts
            print(A)

            #unit vectors
            x = np.array([1, 0, 0]) 
            y = np.array([0, 1, 0])

            #dot products of A with x and y unit vectors to find x and y components of A
            A_x = np.dot(x, A) #the displacement in px corrosponding to 1 step in x
            print ("Step size along X in microns: {}".format(np.linalg.norm(A_x) * microns_per_pixel))
            A_y = np.dot(y, A)
            print ("Step size along Y in microns: {}".format(np.linalg.norm(A_y) * microns_per_pixel))

            #uses standard dot product formula to find angle between A_x and A_y
            dotproduct = np.dot(A_x, A_y)
            cosa = dotproduct / (np.linalg.norm(A_x) * np.linalg.norm(A_y))
            angle = np.arccos(cosa)
            angle = angle * 180 / np.pi
            print ("Angle between axis in degrees: {}".format(angle))

            transformed_stage_positions = np.dot(location_shifts, A)

            numerator = abs(np.linalg.norm(pixel_shifts - transformed_stage_positions, axis = 1)) ** 2
            denominator = abs(np.linalg.norm(pixel_shifts, axis = 1)) ** 2
            error = np.mean(numerator / denominator)
            print("Error in transformation matrix: {}".format(error))

            matplotlib.rcParams.update({'font.size': 12})

            fig, ax = plt.subplots(1, 1)

            ax.plot(transformed_stage_positions[:, 0] * microns_per_pixel, transformed_stage_positions[:, 1] * microns_per_pixel, 'r+')
            ax.plot(pixel_shifts[:, 0] * microns_per_pixel,pixel_shifts[:, 1] * microns_per_pixel, 'b+')

            plt.tight_layout()

            pdf.savefig(fig)
            plt.show()
            plt.close(fig)

            plotting1(transformed_stage_positions[:, 0] * microns_per_pixel,
                      transformed_stage_positions[:, 1] * microns_per_pixel,
                      r"Transformed Stage X Position [$\mathrm{\mu m}$]", r"Transformed Stage Y Position [$\mathrm{\mu m}$]")
            for ia, na in enumerate(["X", "Y"]):
                plotting1(pixel_shifts[:, ia] * microns_per_pixel,
                          transformed_stage_positions[:, ia] * microns_per_pixel,
                          "Camera " + na + r" Position [$\mathrm{\mu m}$]", "Transformed Stage " + na + r" Position [$\mathrm{\mu m}$]")
            #for ia, na in enumerate(["X", "Y"]):
            #    for ib, nb in enumerate(["X", "Y"]):
            #        plotting2(pixel_shifts[:, ia] - np.mean(pixel_shifts[:, ia]),
            #                  transformed_stage_positions[:, ib] - pixel_shifts[:, ib],
            #                  "Camera " + na + " Position [px]", "Error in " + nb + " [px]", group)

            fig1, ax1 = plt.subplots(1, 1)
            ax1.plot(pixel_shifts[:, 0] / np.linalg.norm(A_x),
                    (transformed_stage_positions[:, 0] - pixel_shifts[:, 0]) * microns_per_pixel, ".")
            ax1.set_xlabel(r"Camera X Position [$\mathrm{Steps}$]")
            ax1.set_ylabel(r"Error in X [$\mathrm{\mu m}$]")

            plt.tight_layout()

            pdf.savefig(fig1)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(1, 1)
            ax2.plot(pixel_shifts[:, 0] / np.linalg.norm(A_x),
                     (transformed_stage_positions[:, 0] - pixel_shifts[:, 0]) * microns_per_pixel, ".-")
            ax2.set_xlabel(r"Camera X Position [$\mathrm{Steps}$]")
            ax2.set_ylabel(r"Error in X [$\mathrm{\mu m}$]")

            plt.tight_layout()

            pdf.savefig(fig2)
            plt.close(fig2)

            fig3, ax3 = plt.subplots(1, 1)
            ax3.plot((transformed_stage_positions[:, 0] - pixel_shifts[:, 0]) * microns_per_pixel,
                     (transformed_stage_positions[:, 1] - pixel_shifts[:, 1]) * microns_per_pixel, "r.")

            ax3.set_aspect(1)
            xmin, xmax = ax3.get_xlim()
            ymin, ymax = ax3.get_ylim()
            r = max(abs(xmin), abs(ymin), abs(xmax), abs(ymax)) * 1.1
            ax3.set_xlim(-r, r)
            ax3.set_ylim(-r, r)
            ax3.spines['left'].set_position('zero')
            ax3.spines['right'].set_color('none')
            ax3.spines['bottom'].set_position('zero')
            ax3.spines['top'].set_color('none')
            ax3.set_xlabel(r"Error in X [$\mathrm{\mu m}$]", horizontalalignment = 'right', x = 1.0)
            ax3.set_ylabel(r"Error in Y [$\mathrm{\mu m}$]", horizontalalignment = 'right', y = 1.0)

            plt.tight_layout()

            pdf.savefig(fig3)
            plt.close(fig3)
            
    df.close()
