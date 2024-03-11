def save_frame(i):
    # Change rotation
    ax.view_init(0, i, 180, vertical_axis='y')
    ax.dist = 2.5
    plt.savefig(r"C:\Users\ag-bahl\Desktop\zbrain_mesh\temp_img\frame_{0}.png".format(i), dpi=300)
    return np.array(Image.open(r"C:\Users\ag-bahl\Desktop\zbrain_mesh\temp_img\frame_{0}.png".format(i)))
