# entry point for the GUI app
from .ui.app import PCAApp
import tkinter as tk

def main():
    root = tk.Tk()
    # set a good window size
    root.geometry("1100x700")
    app = PCAApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
