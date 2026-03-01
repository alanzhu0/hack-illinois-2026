from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import include, path


urlpatterns = [
	path("admin/", admin.site.urls),
	path("", include("markets.urls")),
]

if settings.DEBUG:
	urlpatterns += staticfiles_urlpatterns()
	urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
