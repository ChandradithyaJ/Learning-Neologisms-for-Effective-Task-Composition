import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import json
from datetime import datetime
import random

class ImageComparisonGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Comparison Game")
        self.root.geometry("1400x900")
        
        # Game state
        self.base_folder = ""
        self.folder1_path = ""
        self.folder2_path = ""
        self.folder3_path = ""
        self.total_images = 0
        self.current_index = 1
        
        # Scores
        self.scores = {
            'folder1': 0,
            'folder2': 0,
            'equal': 0
        }
        
        # Choices history
        self.choices = []
        
        # Track which folder is on which side (randomized per image)
        self.current_left_folder = None  # will be 'folder1' or 'folder2'
        self.current_right_folder = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Setup frame
        self.setup_frame = ttk.Frame(self.root, padding="10")
        self.setup_frame.pack(fill=tk.X)
        
        # Folder inputs
        ttk.Label(self.setup_frame, text="Base Image Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.base_entry = ttk.Entry(self.setup_frame, width=50)
        self.base_entry.grid(row=0, column=1, padx=5)
        ttk.Button(self.setup_frame, text="Browse", command=lambda: self.browse_folder('base')).grid(row=0, column=2)
        
        ttk.Label(self.setup_frame, text="Folder 1 (Left):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.folder1_entry = ttk.Entry(self.setup_frame, width=50)
        self.folder1_entry.grid(row=1, column=1, padx=5)
        ttk.Button(self.setup_frame, text="Browse", command=lambda: self.browse_folder('folder1')).grid(row=1, column=2)
        
        ttk.Label(self.setup_frame, text="Folder 2 (Right):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.folder2_entry = ttk.Entry(self.setup_frame, width=50)
        self.folder2_entry.grid(row=2, column=1, padx=5)
        ttk.Button(self.setup_frame, text="Browse", command=lambda: self.browse_folder('folder2')).grid(row=2, column=2)
        
        ttk.Label(self.setup_frame, text="Prompts Folder:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.folder3_entry = ttk.Entry(self.setup_frame, width=50)
        self.folder3_entry.grid(row=3, column=1, padx=5)
        ttk.Button(self.setup_frame, text="Browse", command=lambda: self.browse_folder('folder3')).grid(row=3, column=2)
        
        ttk.Label(self.setup_frame, text="Total Images:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.total_entry = ttk.Entry(self.setup_frame, width=20)
        self.total_entry.grid(row=4, column=1, sticky=tk.W, padx=5)
        
        ttk.Button(self.setup_frame, text="Start Game", command=self.start_game).grid(row=5, column=1, pady=10)
        
        # Create a canvas with scrollbar for the game area
        canvas_container = ttk.Frame(self.root)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        
        self.game_frame = ttk.Frame(self.canvas, padding="10")
        
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.game_frame, anchor="nw")
        
        # Bind canvas to update scroll region
        self.game_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # Bind mousewheel for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        
        # Score display
        score_frame = ttk.Frame(self.game_frame)
        score_frame.pack(fill=tk.X, pady=10)
        
        self.score_label = ttk.Label(score_frame, text="Folder 1: 0 pts | Equal: 0 pts | Folder 2: 0 pts", font=('Arial', 14, 'bold'))
        self.score_label.pack()
        
        self.progress_label = ttk.Label(score_frame, text="Image 0 / 0", font=('Arial', 12))
        self.progress_label.pack()
        
        # Images frame
        images_frame = ttk.Frame(self.game_frame)
        images_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Base image
        base_frame = ttk.LabelFrame(images_frame, text="Base Image", padding="10")
        base_frame.grid(row=0, column=0, columnspan=2, pady=10)
        self.base_label = ttk.Label(base_frame)
        self.base_label.pack()
        
        # Prompt
        self.prompt_label = ttk.Label(images_frame, text="", font=('Arial', 12), wraplength=1200, justify=tk.CENTER)
        self.prompt_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Left image
        left_frame = ttk.LabelFrame(images_frame, text="Left Image", padding="10")
        left_frame.grid(row=2, column=0, padx=20)
        self.left_label = ttk.Label(left_frame)
        self.left_label.pack()
        
        # Right image
        right_frame = ttk.LabelFrame(images_frame, text="Right Image", padding="10")
        right_frame.grid(row=2, column=1, padx=20)
        self.right_label = ttk.Label(right_frame)
        self.right_label.pack()
        
        # Buttons
        button_frame = ttk.Frame(self.game_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Choose Left", command=lambda: self.make_choice('left'), width=20).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Equal", command=lambda: self.make_choice('equal'), width=20).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Choose Right", command=lambda: self.make_choice('right'), width=20).pack(side=tk.LEFT, padx=10)
        
        # Export button
        ttk.Button(self.game_frame, text="Export Results", command=self.export_results).pack(pady=10)
        
        # Update canvas width when window is resized
        self.canvas.bind('<Configure>', self._on_canvas_configure)
    
    def _on_mousewheel(self, event):
        # Handle mousewheel scrolling
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
    
    def _on_canvas_configure(self, event):
        # Update the canvas window width to match the canvas width
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        
    def browse_folder(self, folder_type):
        folder = filedialog.askdirectory()
        if folder:
            if folder_type == 'base':
                self.base_entry.delete(0, tk.END)
                self.base_entry.insert(0, folder)
            elif folder_type == 'folder1':
                self.folder1_entry.delete(0, tk.END)
                self.folder1_entry.insert(0, folder)
            elif folder_type == 'folder2':
                self.folder2_entry.delete(0, tk.END)
                self.folder2_entry.insert(0, folder)
            elif folder_type == 'folder3':
                self.folder3_entry.delete(0, tk.END)
                self.folder3_entry.insert(0, folder)
    
    def start_game(self):
        self.base_folder = self.base_entry.get()
        self.folder1_path = self.folder1_entry.get()
        self.folder2_path = self.folder2_entry.get()
        self.folder3_path = self.folder3_entry.get()
        
        try:
            self.total_images = int(self.total_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for total images")
            return
        
        if not all([self.base_folder, self.folder1_path, self.folder2_path, self.folder3_path, self.total_images > 0]):
            messagebox.showerror("Error", "Please fill in all fields")
            return
        
        # Hide the setup frame to make room for images
        self.setup_frame.pack_forget()
        
        # Reset game state
        self.current_index = 1
        self.scores = {'folder1': 0, 'folder2': 0, 'equal': 0}
        self.choices = []
        
        # Load first set
        self.load_current_set()
    
    def load_current_set(self):
        # Update progress
        self.progress_label.config(text=f"Image {self.current_index} / {self.total_images}")
        
        # Load base image
        base_path = os.path.join(self.base_folder, f"{self.current_index}.png")
        self.load_image(base_path, self.base_label, max_size=(300, 200))
        
        # Randomize which folder appears on which side
        if random.random() < 0.5:
            self.current_left_folder = 'folder1'
            self.current_right_folder = 'folder2'
            left_path = os.path.join(self.folder1_path, f"{self.current_index}.png")
            right_path = os.path.join(self.folder2_path, f"{self.current_index}.png")
        else:
            self.current_left_folder = 'folder2'
            self.current_right_folder = 'folder1'
            left_path = os.path.join(self.folder2_path, f"{self.current_index}.png")
            right_path = os.path.join(self.folder1_path, f"{self.current_index}.png")
        
        # Load left and right images
        self.load_image(left_path, self.left_label, max_size=(400, 300))
        self.load_image(right_path, self.right_label, max_size=(400, 300))
        
        # Load prompt
        prompt_path = os.path.join(self.folder3_path, f"{self.current_index}.txt")
        try:
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
                self.prompt_label.config(text=f"Prompt: {prompt}")
        except FileNotFoundError:
            self.prompt_label.config(text=f"Prompt file not found for image {self.current_index}")
        
        # Update scores
        self.update_score_display()
    
    def load_image(self, path, label, max_size=(500, 400)):
        try:
            img = Image.open(path)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label.config(image=photo)
            label.image = photo  # Keep a reference
        except FileNotFoundError:
            label.config(text=f"Image not found: {path}", image='')
    
    def make_choice(self, choice):
        # Determine which folder was actually chosen based on randomization
        if choice == 'left':
            actual_folder = self.current_left_folder
            if actual_folder == 'folder1':
                self.scores['folder1'] += 1
                chosen_image = os.path.join(self.folder1_path, f"{self.current_index}.png")
            else:
                self.scores['folder2'] += 1
                chosen_image = os.path.join(self.folder2_path, f"{self.current_index}.png")
        elif choice == 'right':
            actual_folder = self.current_right_folder
            if actual_folder == 'folder1':
                self.scores['folder1'] += 1
                chosen_image = os.path.join(self.folder1_path, f"{self.current_index}.png")
            else:
                self.scores['folder2'] += 1
                chosen_image = os.path.join(self.folder2_path, f"{self.current_index}.png")
        else:  # equal
            self.scores['equal'] += 1
            chosen_image = 'equal'
            actual_folder = 'equal'
        
        # Record choice
        self.choices.append({
            'index': self.current_index,
            'choice': choice,  # left, right, or equal
            'actual_folder': actual_folder,  # which folder was actually chosen
            'left_was': self.current_left_folder,  # which folder was on the left
            'right_was': self.current_right_folder,  # which folder was on the right
            'chosen_image': chosen_image,
            'timestamp': datetime.now().isoformat()
        })
        
        # Move to next image
        if self.current_index < self.total_images:
            self.current_index += 1
            self.load_current_set()
        else:
            messagebox.showinfo("Complete", f"Game completed!\n\nFinal Scores:\nFolder 1: {self.scores['folder1']} pts\nEqual: {self.scores['equal']} pts\nFolder 2: {self.scores['folder2']} pts")
    
    def update_score_display(self):
        self.score_label.config(text=f"Folder 1: {self.scores['folder1']} pts | Equal: {self.scores['equal']} pts | Folder 2: {self.scores['folder2']} pts")
    
    def export_results(self):
        if not self.choices:
            messagebox.showwarning("No Data", "No choices have been made yet")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            results = {
                'scores': self.scores,
                'choices': self.choices,
                'folders': {
                    'base': self.base_folder,
                    'folder1': self.folder1_path,
                    'folder2': self.folder2_path,
                    'folder3': self.folder3_path
                },
                'total_images': self.total_images
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            messagebox.showinfo("Success", f"Results exported to {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonGame(root)
    root.mainloop()