import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import torch
import numpy as np
import cv2
from queue import Queue
from tkinter import IntVar


class interaction_control:
    def __init__(self, sam=None, image=None):
        self.sam = sam
        self.image = (
            image if image is not None else np.zeros((512, 512, 3), dtype=np.uint8)
        )
        self.boxes = []
        self.masks = None
        self.tk_root = None
        self.image_queue = Queue()

    def on_click(self, event=None):
        # print(f"x: {event.x}, y: {event.y}")
        self.init_x = event.x
        self.init_y = event.y

    def on_release(self, event=None):
        self.end_x = event.x
        self.end_y = event.y
        # print(f"started at x: {self.init_x}, y: {self.init_x}")
        # print(f"released at x: {event.x}, y: {event.y}")
        self.draw_box()
        self.push_box()

    def on_close(self, event=None):
        self.predict_by_boxes()
        self.boxes = []
        self.root.quit()

    def push_box(self):
        self.boxes.append([self.init_x, self.init_y, self.end_x, self.end_y])

    def draw_box(self, event=None):
        rect_d = ImageDraw.Draw(self.image)
        rect_d.rectangle([(self.init_x, self.init_y), (self.end_x, self.end_y)])
        self.photo = ImageTk.PhotoImage(self.image)
        self.l.configure(image=self.photo)

    def predict_by_boxes(self):
        image_np = np.array(self.image)
        self.sam.set_image(image_np)
        if len(self.boxes) == 1:
            trans_boxes = np.array(self.boxes[0])
            masks, _, _ = self.sam.predict(
                point_coords=None,
                point_labels=None,
                box=trans_boxes[None, :],
                multimask_output=False,
            )
            masks = torch.from_numpy(masks).unsqueeze(0)

        elif len(self.boxes) > 1:
            maskss = []
            for i, i_box in enumerate(self.boxes):
                # import pdb; pdb.set_trace()
                self.sam.set_image(image_np)
                trans_boxes = np.array(i_box)
                masks, _, _ = self.sam.predict(
                    point_coords=None,  # np.array([[int(trans_boxes[2]-trans_boxes[0]), int(trans_boxes[3]-trans_boxes[1])]]),
                    point_labels=None,  # np.array([1]),
                    box=trans_boxes[None, :],
                    multimask_output=True,
                )
                print(f"masks shape {masks.shape}")
                maskss.append(torch.from_numpy(masks[0]).unsqueeze(0))
                cv2.imwrite(f"init_mask_{i}.png", masks[0] * 255)
            masks = torch.stack(maskss, dim=0)
        else:
            raise ValueError("No boxes drawn")

        self.masks = masks.to(self.sam.device)
        return self.masks

    def do_interact(self, image=None, tk_root=None):
        if tk_root:  # If a new Tk root is provided
            self.tk_root = tk_root  # Update the stored Tk root

        if not self.tk_root:
            raise ValueError("Tk root must be initialized.")

        if image is not None:
            self.image = image
        elif not self.image_queue.empty():
            self.image = self.image_queue.get()

        if self.image is not None:
            self.photo = ImageTk.PhotoImage(self.image)
            self.l = tk.Label(self.tk_root, image=self.photo)
            self.l.bind("<Button 1>", self.on_click)
            self.l.bind("<ButtonRelease 1>", self.on_release)
            self.l.pack()
            self.l.configure(image=self.photo)

        self.enter_pressed = IntVar(value=0)
        self.tk_root.bind("<Return>", lambda event: self.enter_pressed.set(1))

        # Wait until Enter is pressed (i.e., enter_pressed becomes 1)
        self.tk_root.wait_variable(self.enter_pressed)

        # Now you can call your predict_by_boxes or whatever you want to do when Enter is pressed
        masks = self.predict_by_boxes()

        return masks

    def on_enter(self, event=None):
        self.predict_by_boxes()
        self.boxes = []
        self.tk_root.quit()
