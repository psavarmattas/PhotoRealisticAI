import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'PhotoRealistic AI';
  imageSrc: string | null = null;

  // Handle button click to load a new image from the API
  loadImage() {
    this.imageSrc = 'http://127.0.0.1:5000/generate_image'; // Replace with your API URL
  }
}
