import { ChangeDetectorRef, Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { ButtonModule } from 'primeng/button';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule, HttpClientModule, ButtonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
  providers: [],
})
export class AppComponent {
  constructor(public httpservice: HttpClient, public cdr: ChangeDetectorRef) {}
  title = 'PhotoRealistic AI';
  imageSrc: any = null;
  messages: any = null;

  // Handle button click to load a new image from the API
  loadImage() {
    this.httpservice.get('http://127.0.0.1:5000/generate_image').subscribe(
      (res: any) => {
        this.imageSrc = 'data:image/png;base64,' + res['image'];
        this.cdr.detectChanges();
        this.cdr.markForCheck();
      },
      (error: any) => console.log(error)
    );
  }
}
