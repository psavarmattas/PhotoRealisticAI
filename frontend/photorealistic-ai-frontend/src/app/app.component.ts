import { ChangeDetectorRef, Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { error } from 'console';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule, HttpClientModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
  providers: [],
})
export class AppComponent {
  constructor(public httpservice: HttpClient,public cdr: ChangeDetectorRef) {}
  title = 'PhotoRealistic AI';
  imageSrc: any = null;

  // Handle button click to load a new image from the API
  loadImage() {
    this.httpservice.get("http://127.0.0.1:5000/generate_image").subscribe((res:any)=>{
      this.imageSrc = 'data:image/png;base64,' + res['image'];
      // this.imageSrc = atob(res['image']);÷
      this.cdr.detectChanges();
      this.cdr.markForCheck();
    },
    (error:any)=>console.log(error));
  }
}
