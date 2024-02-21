  raw_depth = Image.fromarray(depth.cpu().numpy().astype('uint16'))
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp.name)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]

        return [(original_image, colored_depth), tmp.name]