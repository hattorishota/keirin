/*----------------------------------------------------
script.js
----------------------------------------------------*/
// jQuery
$(function () {
  // ページ内スクロール
  $('a[href*="#"]').click(function () {
    const speed = 400;
    const target = $(this.hash === '#' || '' ? 'html' : this.hash)
    if (!target.length) return;
    const targetPos = target.offset().top;
    $('html, body').animate({ scrollTop: targetPos }, speed, 'swing');
    return false;
  });

  // Slick
  $('.fv__slider').slick({
    autoplay: true,
    autoplaySpeed: 4000,
    arrows: true,
    speed: 1000,
    fade: true,
    cssEase: 'linear',
  });

  // タブ
  $('.business__tabItem').click(function(){
    $('.current').removeClass('current');
    $(this).addClass('current');
    $('.show').removeClass('show');
    const index = $(this).index();
    $('.business__tabContent').eq(index).addClass('show');
  });
});