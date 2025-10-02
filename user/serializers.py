from rest_framework import serializers
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from .models import CustomUser

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    confirm_password = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = CustomUser
        fields = ('email', 'phone_number', 'business_name', 'province', 'password', 'confirm_password')
        extra_kwargs = {
            'email': {'required': True},
            'phone_number': {'required': True},
            'business_name': {'required': True},
            'province': {'required': True},
        }

    def validate(self, attrs):
        if attrs['password'] != attrs['confirm_password']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        
        # Check if email already exists
        if CustomUser.objects.filter(email=attrs['email']).exists():
            raise serializers.ValidationError({"email": "A user with this email already exists."})
        
        # Check if phone number already exists
        if CustomUser.objects.filter(phone_number=attrs['phone_number']).exists():
            raise serializers.ValidationError({"phone_number": "A user with this phone number already exists."})
        
        return attrs

    def create(self, validated_data):
        validated_data.pop('confirm_password')
        user = CustomUser.objects.create_user(**validated_data)
        return user

class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(write_only=True, required=True)

    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')

        if email and password:
            user = authenticate(request=self.context.get('request'), email=email, password=password)
            
            if not user:
                raise serializers.ValidationError('Invalid credentials. Please try again.')
            
            if not user.is_active:
                raise serializers.ValidationError('Account disabled. Please contact support.')
            
            attrs['user'] = user
            return attrs
        
        raise serializers.ValidationError('Both email and password are required.')

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ('id', 'email', 'phone_number', 'business_name', 'province', 'date_joined')
        read_only_fields = ('id', 'date_joined')

class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for viewing and updating user profile"""
    province_display = serializers.CharField(source='get_province_display', read_only=True)
    
    class Meta:
        model = CustomUser
        fields = ['id', 'email', 'phone_number', 'business_name', 'province', 
                  'province_display', 'date_joined', 'last_login']
        read_only_fields = ['id', 'email', 'date_joined', 'last_login']
    
    def update(self, instance, validated_data):
        """Update user profile fields"""
        instance.phone_number = validated_data.get('phone_number', instance.phone_number)
        instance.business_name = validated_data.get('business_name', instance.business_name)
        instance.province = validated_data.get('province', instance.province)
        instance.save()
        return instance


class ChangePasswordSerializer(serializers.Serializer):
    """Serializer for password change endpoint"""
    current_password = serializers.CharField(required=True, write_only=True)
    new_password = serializers.CharField(required=True, write_only=True)
    confirm_password = serializers.CharField(required=True, write_only=True)
    
    def validate_current_password(self, value):
        """Check that current password is correct"""
        user = self.context['request'].user
        if not user.check_password(value):
            raise serializers.ValidationError("Current password is incorrect.")
        return value
    
    def validate(self, data):
        """Check that new passwords match and meet requirements"""
        if data['new_password'] != data['confirm_password']:
            raise serializers.ValidationError({
                "confirm_password": "New passwords do not match."
            })
        
        # Validate password strength
        try:
            validate_password(data['new_password'], self.context['request'].user)
        except ValidationError as e:
            raise serializers.ValidationError({
                "new_password": list(e.messages)
            })
        
        return data
    
    def save(self):
        """Update user password"""
        user = self.context['request'].user
        user.set_password(self.validated_data['new_password'])
        user.save()
        return user