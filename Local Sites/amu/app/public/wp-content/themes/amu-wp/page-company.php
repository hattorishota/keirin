<!DOCTYPE html>
<html lang="ja">
  <head>
    <?php
    /**
    * The main template file
    *
    */
      get_header();
    ?>
  </head>
  <body>
    <div class="wrapper">
    <?php get_header('tmp'); ?>
    <main class="main under">
        <?php get_header('list'); ?>
        <section class="section company">
            <div class="company__inner">
                <div class="leading__box">
                    <h2 class="leading__en">Company</h2>
                    <p class="leading__jp">会社概要</p>
                </div>
                <ul class="company__list">
                    <li class="company__item">
                        <div class="company__subject">
                            <p class="company__subjectTxt">企業名</p>
                        </div>
                        <div class="company__content">
                            <p class="company__contentTxt">株式会社AMU冷熱</p>
                        </div>
                    </li>
                    <li class="company__item">
                        <div class="company__subject">
                            <p class="company__subjectTxt">代表者</p>
                        </div>
                        <div class="company__content">
                            <p class="company__contentTxt">田沼 克之</p>
                        </div>
                    </li>
                    <li class="company__item">
                        <div class="company__subject">
                            <p class="company__subjectTxt">設立</p>
                        </div>
                        <div class="company__content">
                            <p class="company__contentTxt">平成17年12月</p>
                        </div>
                    </li>
                    <li class="company__item">
                        <div class="company__subject">
                            <p class="company__subjectTxt">資本金</p>
                        </div>
                        <div class="company__content">
                            <p class="company__contentTxt">1000万円</p>
                        </div>
                    </li>
                    <li class="company__item">
                        <div class="company__subject">
                            <p class="company__subjectTxt">所在地</p>
                        </div>
                        <div class="company__content">
                            <p class="company__contentTxt">
                                〒340-0002  <br class="sp-only">埼玉県草加市青柳7-24-1<br>
                                TEL：048-950-8446 <br class="sp-only"><span class="pc-only">/</span> FAX：048-950-8447
                            </p>
                            <div class="company__map">
                                <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3233.6641157207673!2d139.8189434!3d35.8572388!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x6018972c7586c201%3A0xd552cc7c400bf5cf!2z44CSMzQwLTAwMDIg5Z-8546J55yM6I2J5Yqg5biC6Z2S5p-z77yX5LiB55uu77yS77yU4oiS77yR!5e0!3m2!1sja!2sjp!4v1703602349786!5m2!1sja!2sjp" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
                            </div>
                        </div>
                    </li>
                    <li class="company__item">
                        <div class="company__subject">
                            <p class="company__subjectTxt">事業内容</p>
                        </div>
                        <div class="company__content">
                            <p class="company__contentTxt">冷却水循環装置（チラー・サーキュレーター）の製造・販売</p>
                        </div>
                    </li>
                    <li class="company__item">
                        <div class="company__subject">
                            <p class="company__subjectTxt">取引銀行</p>
                        </div>
                        <div class="company__content">
                            <p class="company__contentTxt">
                                りそな銀行<br>
                                栃木銀行
                            </p>
                        </div>
                    </li>
                </ul>
            </div>
        </section>
        <?php get_footer(); ?>