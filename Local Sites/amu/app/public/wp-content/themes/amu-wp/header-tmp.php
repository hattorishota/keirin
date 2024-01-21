<header class="header">
        <div class="header__wrap">
          <div class="header__logo">
            <p class="header__logoTxt">チラー(冷却水循環装置)の企画設計・製造販売・OEM提供</p>
            <a href="<?php echo home_url(); ?>">
              <img src="<?php echo get_template_directory_uri(); ?>/img/common/logo.png" alt="amu冷熱" />
            </a>
          </div>
          <div class="header__menuWrap">
            <input type="checkbox" id="menuCheck" class="header__input" />
            <label for="menuCheck" class="header__menuBtn">
              <span></span>
            </label>
            <div class="header__menuContent">
              <div class="header__menuContentInner">
                <ul class="header__list">
                  <li class="header__item">
                    <a href="<?php echo home_url(); ?>" class="header__item--flex">
                      <span class="header__item--ja">ホーム</span>
                      <span class="header__item--en">Home</span>
                    </a>
                  </li>
                  <li class="header__item">
                    <a href="<?php echo home_url('about'); ?>" class="header__item--flex">
                      <span class="header__item--ja">事業案内</span>
                      <span class="header__item--en">Business</span>
                    </a>
                  </li>
                  <li class="header__item">
                    <a href="<?php echo home_url('company'); ?>" class="header__item--flex">
                      <span class="header__item--ja">会社概要</span>
                      <span class="header__item--en">Company</span>
                    </a>
                  </li>
                  <li class="header__item">
                    <a href="<?php echo home_url('news'); ?>" class="header__item--flex">
                      <span class="header__item--ja">お知らせ＆ブログ</span>
                      <span class="header__item--en">News & Blog</span>
                    </a>
                  </li>
                  <li class="header__item header__item--tel">
                    <div class="header__spMenuLeadingBox">
                      <p class="header__spMenuLeading">Contact</p>
                      <p class="header__spMenuSubLeading">お問い合わせ</p>
                    </div>
                    <p class="header__spMenuTxt">ご質問などありましたら、お電話・お問い合わせフォームよりお問い合わせください。</p>
                    <div class="header__spMenuTelBox">
                      <p class="header__spMenuTelTxt"></p>
                      <div class="header__spMenuTelNumberBox">
                        <a href="tel:048-950-8446" class="header__spMenuTelNumber">048-950-8446</a>
                        <p class="header__spMenuTelHour">受付時間：平日10:00～17:00</p>
                      </div>
                    </div>
                  </li>
                  <li class="header__item header__item--reserve">
                    <a href="<?php echo home_url('contact'); ?>">お問い合わせ</a>
                  </li>
                </ul>
                <div class="header__spMenu">
                  <div class="header__spMenuButtonBox">
                    <a href="<?php echo home_url('contact'); ?>" class="btn">お問い合わせフォームはこちら</a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>