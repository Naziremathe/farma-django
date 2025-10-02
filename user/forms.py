from django import forms
from .models import CropListing
from datetime import date


class CropListingForm(forms.ModelForm):
    class Meta:
        model = CropListing
        fields = [
            'category', 
            'crop_name', 
            'crop_description', 
            'image', 
            'quantity', 
            'sell_by_date', 
            'price_per_kg'
        ]

    category = forms.ChoiceField(
        choices=CropListing.CATEGORY_CHOICES,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors',
            'id': 'category'
        })
    )

    crop_name = forms.ChoiceField(
        choices=[],  # will be set in __init__
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors',
            'id': 'crop_name'
        })
    )

    crop_description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors',
            'rows': 3,
            'placeholder': 'Provide additional details about the crop (optional)',
            'id': 'crop_description'
        })
    )

    image = forms.ImageField(
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors',
            'accept': 'image/*',
            'id': 'id_image'
        })
    )

    quantity = forms.DecimalField(
        max_digits=10,
        decimal_places=2,
        min_value=0.01,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors',
            'placeholder': 'Enter quantity in kilograms',
            'step': '0.01',
            'id': 'quantity'
        })
    )

    sell_by_date = forms.DateField(
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors',
            'id': 'sell_by_date'
        })
    )

    price_per_kg = forms.DecimalField(
        max_digits=8,
        decimal_places=2,
        min_value=0.01,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors',
            'placeholder': 'Select a crop first to see price suggestions',
            'step': '0.01',
            'id': 'price_per_kg'
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Default empty
        self.fields['crop_name'].choices = [('', 'Select category first')]

        category = self.data.get('category') or self.initial.get('category')
        if category == 'fruits':
            self.fields['crop_name'].choices = CropListing.FRUIT_CHOICES
        elif category == 'vegetables':
            self.fields['crop_name'].choices = CropListing.VEGETABLE_CHOICES
        elif category == 'livestock':
            self.fields['crop_name'].choices = CropListing.LIVESTOCK_CHOICES

    def clean_sell_by_date(self):
        sell_date = self.cleaned_data.get('sell_by_date')
        if sell_date and sell_date < date.today():
            raise forms.ValidationError("Sell by date cannot be in the past.")
        return sell_date
