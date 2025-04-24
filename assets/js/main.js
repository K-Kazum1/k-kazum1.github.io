document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      
      document.querySelector(this.getAttribute('href')).scrollIntoView({
        behavior: 'smooth'
      });
    });
  });
  
    
  const navToggle = document.querySelector('.nav-toggle');
  if (navToggle) {
    navToggle.addEventListener('click', () => {
      const nav = document.querySelector('.site-nav');
      nav.classList.toggle('nav-open');
    });
  }
  
  // Yonaka character animation or interactions
  const yonakaElements = document.querySelectorAll('.yonaka-quote');
  if (yonakaElements.length > 0) {
    yonakaElements.forEach(element => {
      // Add hover effects or small animations for Yonaka quotes
      element.addEventListener('mouseenter', () => {
        element.classList.add('yonaka-hover');
      });
      
      element.addEventListener('mouseleave', () => {
        element.classList.remove('yonaka-hover');
      });
    });
  }
});
