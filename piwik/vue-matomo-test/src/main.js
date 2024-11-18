

import Vue from "vue";
import App from "./App.vue";
import router from "./router";
import store from "./store";
import { createApp } from 'vue'
// import App from './App.vue'

import VueMatomo from 'vue-matomo';

Vue.config.productionTip = false;

Vue.use(VueMatomo, {
  host: "https://192.168.0.227:8080",
  siteId: 1,
  trackerFileName: 'toyouAI',
  router: router,
  enableLinkTracking: true,
  requireConsent: false,
  trackInitialView: true,
  disableCookies: false,
  enableHeartBeatTimer: false,
  heartBeatTimerInterval: 15,
  debug: false,
  userId: undefined,
  cookieDomain: undefined,
  domains: undefined,
  preInitActions: []
});

createApp(App).mount('#app')
