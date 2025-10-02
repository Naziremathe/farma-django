from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from phonenumber_field.modelfields import PhoneNumberField
from django.utils import timezone
from django.core.validators import MinValueValidator

class CustomUserManager(BaseUserManager):
    def create_user(self, email, phone_number, business_name, province, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        if not phone_number:
            raise ValueError('The Phone Number field must be set')
        
        email = self.normalize_email(email)
        user = self.model(
            email=email,
            phone_number=phone_number,
            business_name=business_name,
            province=province,
            **extra_fields
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, phone_number, business_name, province, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)

        return self.create_user(email, phone_number, business_name, province, password, **extra_fields)

class CustomUser(AbstractBaseUser, PermissionsMixin):
    PROVINCE_CHOICES = [
        ('EC', 'Eastern Cape'),
        ('FS', 'Free State'),
        ('GP', 'Gauteng'),
        ('KZN', 'KwaZulu-Natal'),
        ('LP', 'Limpopo'),
        ('MP', 'Mpumalanga'),
        ('NC', 'Northern Cape'),
        ('NW', 'North West'),
        ('WC', 'Western Cape'),
    ]

    email = models.EmailField(unique=True, db_index=True)
    phone_number = PhoneNumberField(unique=True, db_index=True)
    business_name = models.CharField(max_length=255, db_index=True)
    province = models.CharField(max_length=3, choices=PROVINCE_CHOICES)
    
    # Django auth fields
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    
    # Timestamps
    date_joined = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(auto_now=True)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['phone_number', 'business_name', 'province']

    class Meta:
        db_table = 'custom_users'
        indexes = [
            models.Index(fields=['email']),
            models.Index(fields=['phone_number']),
            models.Index(fields=['business_name']),
            models.Index(fields=['province']),
        ]

    def __str__(self):
        return self.email

    def get_full_name(self):
        return self.business_name

    def get_short_name(self):
        return self.business_name
    



class CropListing(models.Model):
    CATEGORY_CHOICES = [
        ('fruits', 'Fruits'),
        ('vegetables', 'Vegetables'),
        ('livestock', 'Livestock'),
    ]
    
    FRUIT_CHOICES = [
        ('apples', 'Apples'),
        ('bananas', 'Bananas'),
        ('oranges', 'Oranges'),
        ('grapes', 'Grapes'),
        ('berries', 'Berries'),
        ('mangoes', 'Mangoes'),
        ('pineapples', 'Pineapples'),
        ('watermelons', 'Watermelons'),
        ('other_fruit', 'Other Fruit'),
    ]
    
    VEGETABLE_CHOICES = [
        ('tomatoes', 'Tomatoes'),
        ('potatoes', 'Potatoes'),
        ('carrots', 'Carrots'),
        ('onions', 'Onions'),
        ('cabbage', 'Cabbage'),
        ('spinach', 'Spinach'),
        ('broccoli', 'Broccoli'),
        ('other_vegetable', 'Other Vegetable'),
    ]
    
    LIVESTOCK_CHOICES = [
        ('cattle', 'Cattle'),
        ('poultry', 'Poultry'),
        ('goats', 'Goats'),
        ('sheep', 'Sheep'),
        ('pigs', 'Pigs'),
        ('fish', 'Fish'),
        ('other_livestock', 'Other Livestock'),
    ]

    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('published', 'Published'),
        ('sold', 'Sold'),
        ('expired', 'Expired'),
    ]

    # Basic Information
    farmer = models.ForeignKey('CustomUser', on_delete=models.CASCADE, related_name='crop_listings')
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    crop_name = models.CharField(max_length=50)
    crop_description = models.TextField(blank=True, null=True)
    
    # Image Field
    image = models.ImageField(upload_to='crop_images/', null=True, blank=True)
    
    # Quantity Information
    quantity = models.DecimalField(max_digits=10, decimal_places=2, validators=[MinValueValidator(0.01)])
    unit = models.CharField(max_length=20, default='kg')
    
    # Timing
    sell_by_date = models.DateField()
    
    # Pricing
    price_per_kg = models.DecimalField(max_digits=10, decimal_places=2, validators=[MinValueValidator(0)])
    
    # Status and Metadata
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'crop_listings'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['category']),
            models.Index(fields=['status']),
            models.Index(fields=['sell_by_date']),
        ]

    def __str__(self):
        return f"{self.get_category_display()} - {self.crop_name} - {self.farmer.business_name}"

    @property
    def total_value(self):
        if self.price_per_kg:
            return self.quantity * self.price_per_kg
        return None

    def publish(self):
        self.status = 'published'
        self.published_at = timezone.now()
        self.save()