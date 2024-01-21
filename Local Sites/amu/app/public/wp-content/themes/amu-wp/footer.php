<section class="contact">
          <div class="contact__inner">
            <div class="contact__left">
              <h2 class="contact__title">Contact</h2>
              <p class="contact__subtitle">お問い合わせ</p>
              <p class="contact__txt">
                ご質問などありましたら、<br class="pc-only">
                お電話・お問い合わせフォームより<br class="pc-only">
                お問い合わせください
              </p>
            </div>
            <div class="contact__right">
              <div class="contact__telBox">
                <div class="contact__telTxtBox">
                  <p class="contact__telTxt">TEL</p>
                </div>
                <div class="contact__telNumberBox">
                  <a href="tel:048-950-8446">048-950-8446</a>
                  <p class="contact__telNumberHour">受付時間：平日10:00～17:00</p>
                </div>
              </div>
              <div class="contact__buttonBox">
                <a href="<?php echo home_url('contact'); ?>" class="btn">お問い合わせフォームはこちら</a>
              </div>
            </div>
          </div>
        </section>
      </main>
      <footer class="footer">
        <div class="footer__inner">
          <div class="footer__info">
            <div class="footer__logo">
              <img src="<?php echo get_template_directory_uri(); ?>/img/common/logo.png" alt="">
            </div>
            <p class="footer__txt footer__adress">埼玉県草加市青柳7-24-1</p>
            <p class="footer__txt footer__tel">TEL：048-950-8446  /  FAX：048-950-8447</p>
            <p class="footer__txt footer__hour">受付時間：平日10:00～17:00</p>
          </div>
          <nav class="footer__nav">
            <ul class="footer__list footer__list--mr4">
              <li class="footer__item">
                <a href="<?php echo home_url(); ?>">Home</a>
              </li>
              <li class="footer__item">
                <a href="<?php echo home_url('about'); ?>">事業案内</a>
              </li>
              <li class="footer__item">
                <a href="<?php echo home_url('company'); ?>">会社概要</a>
              </li>
              <li class="footer__item">
                <a href="<?php echo home_url('news'); ?>">お知らせ＆ブログ</a>
              </li>
              <li class="footer__item footer__item--mr0">
                <a href="<?php echo home_url('contact'); ?>">お問い合わせ</a>
              </li>
              <li class="footer__item sp-only">
                <a href="<?php echo home_url('privacypolicy'); ?>">プライバシーポリシー</a>
              </li>
            </ul>
            <ul class="footer__list footer__list--right pc-only">
              <li class="footer__item">
                <a href="<?php echo home_url('privacypolicy'); ?>">プライバシーポリシー</a>
              </li>
            </ul>
          </nav>
        </div>
        <p class="footer__copyright">©AMU Reinetsu Co., Ltd. All Rights Reserved.</p>
      </footer>
    </div>
    <script src="<?php echo get_template_directory_uri(); ?>/js/jquery.min.js"></script>
    <script src="<?php echo get_template_directory_uri(); ?>/js/slick.min.js"></script>
    <script src="<?php echo get_template_directory_uri(); ?>/js/script.js"></script>
    <?php wp_footer(); ?>
  </body>
</html>
