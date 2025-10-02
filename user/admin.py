from django.contrib import admin


from user.models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('email', 'phone_number', 'business_name', 'province', 'is_staff', 'is_active')
    search_fields = ('email', 'phone_number', 'business_name')
    list_filter = ('is_staff', 'is_active', 'province')
    ordering = ('email',)