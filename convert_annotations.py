import os
from PIL import Image

files = os.listdir("ExDark_Annno/People")
for file in files:
    image = os.path.splitext(os.path.splitext(file)[0])[0]
    if os.path.exists(os.path.join("ExDark/People",image+".jpg")):
        image_path = os.path.join("ExDark/People",image+".jpg")
    elif os.path.exists(os.path.join("ExDark/People",image+".png")):
        image_path = os.path.join("ExDark/People",image+".png")
    elif os.path.exists(os.path.join("ExDark/People",image+".jpeg")):
        image_path = os.path.join("ExDark/People",image+".jpeg")
    else:
        print(f"Image file for {image} not found.")
        continue

    img = Image.open(image_path)
    img_w, img_h = img.size
    yolo_lines = []
    annotation_path = os.path.join("ExDark_Annno","People",file)
    with open(annotation_path,"r") as anno:
        for line in anno:
            if line.startswith("%"):
                continue
            parts = line.split()
            if parts[0]!="People":
                continue
            
            x, y, w, h = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

            x_center = (x+w/2)/img_w
            y_center = (y+h/2)/img_h
            w_norm = w/img_w
            h_norm = h/img_h
            res = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(res)
    os.makedirs(os.path.join("dataset","labels","people"),exist_ok=True)
    output_file_path = os.path.join("dataset","labels","people",image+".txt")
    with open(output_file_path,"w") as output_file:
        output_file.write("\n".join(yolo_lines))



